from . import aggregator
from . import affinitymat
from . import extract_seqlets
from . import core
from . import gapped_kmer
from . import io
from . import util
from . import rna
from . import tfmodisco
from . import cluster
from . import progress

__version__ = '2.6.0'


def __getattr__(name):
	"""Load reporting modules only when a reporting command needs one."""
	if name in {"report", "meme_writer", "bed_writer", "fasta_writer",
			"descriptive_report"}:
		import importlib
		module = importlib.import_module("{}.{}".format(__name__, name))
		globals()[name] = module
		return module
	raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
