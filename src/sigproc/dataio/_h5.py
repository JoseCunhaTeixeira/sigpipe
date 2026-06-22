import h5py


def dataset(group: h5py.Group, key: str) -> h5py.Dataset:
    """Fetch `key` as an h5py Dataset, narrowing the Group | Dataset | Datatype union."""
    obj = group[key]
    if not isinstance(obj, h5py.Dataset):
        raise TypeError(f"Expected h5py.Dataset for '{key}', got {type(obj).__name__}")
    return obj
