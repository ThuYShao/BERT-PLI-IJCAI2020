from .nlp.JsonFromFiles import JsonFromFilesDataset
from .others.FilenameOnly import FilenameOnlyDataset

dataset_list = {
    "JsonFromFiles": JsonFromFilesDataset,
    "FilenameOnly": FilenameOnlyDataset
}
