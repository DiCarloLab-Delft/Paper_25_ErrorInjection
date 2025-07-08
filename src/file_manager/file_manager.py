# -------------------------------------------
# Module describing the storage of (request and) response datasets (+ analysis components).
# Keeps track of tuid and meta-description in single overview file.
# -------------------------------------------
import os
from pathlib import Path
from typing import List, Dict, Optional
import xarray as xr
from filelock import FileLock
from quantify_core.data.types import TUID
from quantify_core.data.handling import (
    set_datadir,
    gen_tuid,
    create_exp_folder,
    locate_experiment_container,
    get_latest_tuid,
    get_tuids_containing,
    DATASET_NAME,
    write_dataset,
    load_dataset,
    load_snapshot,
)
from quantify_core.measurement.control import _DATASET_LOCKS_DIR
from qce_circuit.connectivity.intrf_channel_identifier import IQubitID, IEdgeID, DirectedEdgeIDObj

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent.absolute()
DATA_DIR = os.path.join(ROOT_DIR, 'processed_datasets')
FIGURE_DIR = os.path.join(ROOT_DIR, 'data', 'figures')


def enforce_data_directory(func):
    def wrapper(*args, **kwargs):
        set_datadir(StorageManager.data_directory())
        return func(*args, **kwargs)
    return wrapper


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class StorageManager(metaclass=Singleton):
    """
    (Singleton) Behaviour class, describing how data is stored (and retrieved).
    """

    # region Class Properties
    @classmethod
    def data_directory(cls) -> Path:
        return Path(DATA_DIR)

    @classmethod
    def figure_directory(cls) -> Path:
        return Path(FIGURE_DIR)
    # endregion

    # region Class Methods
    @classmethod
    @enforce_data_directory
    def locate_experiment_container(cls, tuid: TUID, data_name: str = "") -> Path:
        """
        Quantify-core wrapper.
        Constructs directory if not already exists.
        """
        try:
            experiment_directory = locate_experiment_container(tuid=tuid)
        except FileNotFoundError:
            experiment_directory = create_exp_folder(tuid=tuid, name=data_name)

        return Path(experiment_directory)

    @classmethod
    @enforce_data_directory
    def create_experiment_container(cls, data_name: str) -> TUID:
        """
        Quantify-core wrapper.
        Constructs directory at TUID-now.
        """
        tuid: TUID = gen_tuid()
        cls.locate_experiment_container(tuid=tuid, data_name=data_name)
        return tuid

    @classmethod
    @enforce_data_directory
    def get_latest_tuid(cls, contains: str = "") -> TUID:
        """
        Quantify-core wrapper.
        """
        return get_latest_tuid(contains=contains)

    @classmethod
    @enforce_data_directory
    def get_tuids(cls, regex: str, start_tuid: TUID, end_tuid: TUID, reverse: bool = False) -> List[TUID]:
        """
        Quantify-core wrapper.
        Collects 'latest' tuid matches and matches to edge-ID.
        """
        return get_tuids_containing(
            contains=regex,
            t_start=TUID.datetime(start_tuid),
            t_stop=TUID.datetime(end_tuid),
            reverse=reverse,
        )

    @classmethod
    @enforce_data_directory
    def get_latest_qubit_to_tuid(cls, qubit_to_regex: Dict[IQubitID, str], start_tuid: TUID, end_tuid: TUID) -> Dict[IQubitID, TUID]:
        """
        Quantify-core wrapper.
        Collects 'latest' tuid matches and matches to edge-ID.
        """
        # Data allocation
        result: Dict[IQubitID, TUID] = {}
        for qubit_id, _contains in qubit_to_regex.items():
            try:
                tuids = get_tuids_containing(
                        contains=_contains,
                        t_start=TUID.datetime(start_tuid),
                        t_stop=TUID.datetime(end_tuid),
                )
                result[qubit_id] = tuids[-1]
            except FileNotFoundError as e:
                pass

        return result

    @classmethod
    @enforce_data_directory
    def get_latest_edge_to_tuid(cls, edges_to_regex: Dict[IEdgeID, str], start_tuid: TUID, end_tuid: TUID) -> Dict[IEdgeID, TUID]:
        """
        Quantify-core wrapper.
        Collects 'latest' tuid matches and matches to edge-ID.
        """
        # Data allocation
        result: Dict[IEdgeID, TUID] = {}
        for edge_id, _contains in edges_to_regex.items():
            try:
                tuids = get_tuids_containing(
                        contains=_contains,
                        t_start=TUID.datetime(start_tuid),
                        t_stop=TUID.datetime(end_tuid),
                )
                result[edge_id] = tuids[-1]
            except FileNotFoundError as e:
                pass

        return result

    @classmethod
    @enforce_data_directory
    def get_latest_directional_edge_to_tuid(cls, edges_to_regex: Dict[DirectedEdgeIDObj, str], start_tuid: TUID, end_tuid: TUID) -> Dict[DirectedEdgeIDObj, TUID]:
        """
        Quantify-core wrapper.
        Collects 'latest' tuid matches and matches to (directional) edge-ID.
        """
        # Data allocation
        result: Dict[DirectedEdgeIDObj, TUID] = {}
        for edge_id, _contains in edges_to_regex.items():
            try:
                tuids = get_tuids_containing(
                        contains=_contains,
                        t_start=TUID.datetime(start_tuid),
                        t_stop=TUID.datetime(end_tuid),
                )
                result[edge_id] = tuids[-1]
            except FileNotFoundError as e:
                pass

        return result

    @classmethod
    def store_data(cls, data: xr.Dataset, tuid: Optional[TUID] = None, data_name: str = "") -> TUID:
        """
        Stores xarray Dataset using quantify-core file management functionalities.
        :param data: Contains dataset to be stored.
        :param tuid: (Optional) Unique time identifier, used for file management during data storing.
        :param data_name: (Optional) name used during file path construction.
        :return: Time identifier at which the data is stored.
        """
        if tuid is None:
            tuid: TUID = gen_tuid()
        dataset = data

        experiment_directory = cls.locate_experiment_container(
            tuid=tuid,
            data_name=data_name,
        )
        dataset_file_path: Path = Path(os.path.join(experiment_directory, DATASET_NAME))

        # Add attributes
        dataset.attrs["name"] = data_name
        dataset.attrs["tuid"] = tuid

        # Multiprocess safe
        lockfile = (
                _DATASET_LOCKS_DIR / f"{tuid}-{DATASET_NAME}.lock"
        )
        with FileLock(lockfile, 5):
            write_dataset(
                path=dataset_file_path,
                dataset=dataset,
            )

        return tuid

    @classmethod
    @enforce_data_directory
    def load_data(cls, tuid: TUID) -> xr.Dataset:
        """
        Loads xarray Dataset using quantify-core file management functionalities.
        :param tuid: Unique time identifier, used for file management during data retrieval.
        :return: xarray dataset.
        """
        dataset = load_dataset(tuid)
        return dataset

    @classmethod
    @enforce_data_directory
    def load_snapshot(cls, tuid: TUID) -> dict:
        """
        Loads json Dataset using quantify-core file management functionalities.
        :param tuid: Unique time identifier, used for file management during data retrieval.
        :return: JSON dictionary containing snapshot dataset.
        """
        return load_snapshot(tuid)
    # endregion
