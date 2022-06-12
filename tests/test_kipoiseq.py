from collections import namedtuple
import unittest
from pkg_resources import resource_filename
from spliceai.utils import Annotator, get_delta_scores

Record = namedtuple('Record', ['chrom', 'pos', 'ref', 'alts'])

from kipoi.data import SampleIterator
from kipoiseq import Interval, Variant
from kipoiseq.transforms import OneHot
from kipoiseq.extractors import VariantSeqExtractor, SingleVariantMatcher, BaseExtractor, FastaStringExtractor
from kipoi.metadata import GenomicRanges

import pandas as pd
import pyranges as pr

from kipoiseq.variant_source import VariantFetcher

# model definition
from kipoi.model import BaseModel
from keras.models import load_model
from pkg_resources import resource_filename
from spliceai.utils import one_hot_encode
import numpy as np


class SpliceAI_model(BaseModel):
    def __init__(self):
        model_paths = ('models/spliceai{}.h5'.format(x) for x in range(1, 6))
        self.models = [load_model(resource_filename('spliceai', x)) for x in model_paths]

    def _predict(self, inputs):
        x = np.asarray([one_hot_encode(input_sequence) for input_sequence in inputs])
        y = np.mean([m.predict(x) for m in self.models], axis=0)

        acceptor_prob = y[:, :, 1]
        donor_prob = y[:, :, 2]

        return acceptor_prob, donor_prob

    def _predict(self, input_sequences):
        x = np.asarray([one_hot_encode(input_sequence) for input_sequence in input_sequences])
        y = np.mean([m.predict(x) for m in self.models], axis=0)

        return {
            "acceptor_prob": y[:, :, 1],
            "donor_prob": y[:, :, 2],
        }

    def predict_on_batch(self, inputs):
        y_ref = self._predict(inputs["ref_seq"])
        y_alt = self._predict(inputs["alt_seq"])

        acceptor_gain: np.ndarray = (y_alt["acceptor_prob"] - y_ref["acceptor_prob"])  # alt - ref
        acceptor_loss = (y_ref["acceptor_prob"] - y_alt["acceptor_prob"])  # ref - alt
        donor_gain = (y_alt["donor_prob"] - y_ref["donor_prob"])  # alt - ref
        donor_loss = (y_ref["donor_prob"] - y_alt["donor_prob"])  # ref - alt

        retval = {
            "DS_AG": acceptor_gain.argmax(axis=-1),
            "DS_AL": acceptor_loss.argmax(axis=-1),
            "DS_DG": donor_gain.argmax(axis=-1),
            "DS_DL": donor_loss.argmax(axis=-1),
            "DP_AG": acceptor_gain.max(axis=-1),
            "DP_AL": acceptor_loss.max(axis=-1),
            "DP_DG": donor_gain.max(axis=-1),
            "DP_DL": donor_loss.max(axis=-1),
        }

        return retval


class SpliceAI_DL(SampleIterator):
    def __init__(
            self,
            reference_sequence: BaseExtractor,
            variants: VariantFetcher,
            distance_around_variant=5000,
            context_width=10000,
            interval_attrs=('gene_id', 'transcript_id')
    ):
        self.reference_sequence = reference_sequence
        self.variants = variants
        self.interval_attrs = interval_attrs

        self.variant_seq_extractor = VariantSeqExtractor(reference_sequence=reference_sequence)

        # self.one_hot = OneHot()

        self.cov = 2 * distance_around_variant + 1
        self.window_width = context_width + self.cov

    def __iter__(self):
        variant: Variant
        strand: str
        for variant in self.variants:
            for strand in ["+", "-"]:
                interval = Interval(
                    chrom=variant.chrom,
                    start=variant.pos - self.window_width // 2 - 1,
                    end=variant.pos + self.window_width // 2,
                    strand=strand
                )
                assert interval.width() == self.window_width, "Invalid window width!"

                yield {
                    "inputs": {
                        "ref_seq": self.reference_sequence.extract(interval),
                        "alt_seq": self.variant_seq_extractor.extract(
                            interval,
                            [variant],
                            anchor=self.window_width // 2,
                        ),
                    },
                    "metadata": {
                        "variant": {
                            "chrom": variant.chrom,
                            "start": variant.start,
                            "end": variant.end,
                            "ref": variant.ref,
                            "alt": variant.alt,
                            "id": variant.id,
                            "str": str(variant),
                        },
                        "ranges": GenomicRanges.from_interval(interval),
                        "strand": interval.strand,
                        **{k: interval.attrs.get(k, '') for k in self.interval_attrs},
                    }
                }


class Kipoi_SpliceAI_DL(SpliceAI_DL):
    def __init__(
            self,
            fasta_file,
            vcf_file,
            vcf_file_tbi=None,
            vcf_lazy=True,
    ):
        from kipoiseq.extractors import MultiSampleVCF
        super().__init__(
            reference_sequence=FastaStringExtractor(fasta_file),
            variants=MultiSampleVCF(vcf_file, lazy=vcf_lazy)
        )


class TestKipoi_SpliceAI_DL(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from pkg_resources import resource_filename

        cls.fasta_path = resource_filename(__name__, 'data/test.fa')
        cls.fasta_without_prefix_path = resource_filename(__name__, 'data/test_without_prefix.fa')

        # cls.ann = Annotator(cls.fasta_path, 'grch37')
        # cls.ann_without_prefix = Annotator(cls.fasta_without_prefix_path, 'grch37')

    def test_get_delta_score_acceptor(self):
        from kipoiseq.extractors.vcf_matching import PyrangesVariantFetcher, Variant

        variant1 = Variant(chrom='10', pos=94077, ref='A', alt='C')
        variant2 = Variant(chrom='10', pos=94077, ref='A', alt='T')
        variant_fetcher = PyrangesVariantFetcher(variants=[variant1, variant2])

        reference_sequence = FastaStringExtractor(self.fasta_without_prefix_path, use_strand=True)

        dl = SpliceAI_DL(
            reference_sequence=reference_sequence,
            variants=variant_fetcher
        )
        input_sample = next(dl.batch_iter(batch_size=4))

        model = SpliceAI_model()

        scores = model.predict_on_batch(input_sample["inputs"])

        scores = get_delta_scores(record, self.ann, 500, 0)
        self.assertEqual(scores, ['C|TUBB8|0.15|0.27|0.00|0.05|89|-23|-267|193'])
        scores = get_delta_scores(record, self.ann_without_prefix, 500, 0)
        self.assertEqual(scores, ['C|TUBB8|0.15|0.27|0.00|0.05|89|-23|-267|193'])

        record = Record('chr10', 94077, 'A', ['C'])
        scores = get_delta_scores(record, self.ann, 500, 0)
        self.assertEqual(scores, ['C|TUBB8|0.15|0.27|0.00|0.05|89|-23|-267|193'])
        scores = get_delta_scores(record, self.ann_without_prefix, 500, 0)
        self.assertEqual(scores, ['C|TUBB8|0.15|0.27|0.00|0.05|89|-23|-267|193'])

    def test_get_delta_score_donor(self):
        record = Record('10', 94555, 'C', ['T'])
        scores = get_delta_scores(record, self.ann, 500, 0)
        self.assertEqual(scores, ['T|TUBB8|0.01|0.18|0.15|0.62|-2|110|-190|0'])
        scores = get_delta_scores(record, self.ann_without_prefix, 500, 0)
        self.assertEqual(scores, ['T|TUBB8|0.01|0.18|0.15|0.62|-2|110|-190|0'])

        record = Record('chr10', 94555, 'C', ['T'])
        scores = get_delta_scores(record, self.ann, 500, 0)
        self.assertEqual(scores, ['T|TUBB8|0.01|0.18|0.15|0.62|-2|110|-190|0'])
        scores = get_delta_scores(record, self.ann_without_prefix, 500, 0)
        self.assertEqual(scores, ['T|TUBB8|0.01|0.18|0.15|0.62|-2|110|-190|0'])
