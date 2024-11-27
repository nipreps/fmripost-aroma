"""BIDS-related interfaces for fMRIPost-template."""

from json import loads

from bids.layout import Config
from niworkflows.interfaces.bids import DerivativesDataSink as BaseDerivativesDataSink

from fmripost_template.data import load as load_data

# NOTE: Modified for fmripost_template's purposes
fmripost_template_spec = loads(load_data('io_spec.json').read_text())
bids_config = Config.load('bids')
deriv_config = Config.load('derivatives')

fmripost_template_entities = {v['name']: v['pattern'] for v in fmripost_template_spec['entities']}
merged_entities = {**bids_config.entities, **deriv_config.entities}
merged_entities = {k: v.pattern for k, v in merged_entities.items()}
merged_entities = {**merged_entities, **fmripost_template_entities}
merged_entities = [{'name': k, 'pattern': v} for k, v in merged_entities.items()]
config_entities = frozenset({e['name'] for e in merged_entities})


class DerivativesDataSink(BaseDerivativesDataSink):
    """Store derivative files.

    A child class of the niworkflows DerivativesDataSink,
    using fmripost_template's configuration files.
    """

    out_path_base = ''
    _allowed_entities = set(config_entities)
    _config_entities = config_entities
    _config_entities_dict = merged_entities
    _file_patterns = fmripost_template_spec['default_path_patterns']
