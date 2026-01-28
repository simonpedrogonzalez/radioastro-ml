import os
import hashlib
import shutil
from casatools import table
import numpy as np

def col_diff(ms1, ms2, col="DATA"):
    from casatools import table
    import numpy as np

    tb = table()
    tb.open(ms1)
    d0 = tb.getcol(col)
    tb.close()

    tb = table()
    tb.open(ms2)
    d1 = tb.getcol(col)
    tb.close()

    print(f"||{col}(before) - {col}(after)|| =", np.linalg.norm(d0 - d1))

def hash_casa_table_cols(
    tabpath: str,
    cols: list[str],
    *,
    row_stride: int = 1,
    max_rows: int | None = None,
) -> str:
    """
    Hash selected columns of a CASA table.

    Parameters
    ----------
    tabpath : str
        Path to CASA table (MS main table or caltable directory)
    cols : list of str
        Column names to hash (must exist in the table)
    row_stride : int
        Hash every k-th row (1 = all rows)
    max_rows : int or None
        Cap number of rows hashed (after striding)

    Returns
    -------
    sha256 hex digest
    """
    tb = table()
    tb.open(tabpath)

    h = hashlib.sha256()
    h.update(tabpath.encode())

    nrows = tb.nrows()
    rows = np.arange(0, nrows, row_stride)
    if max_rows is not None:
        rows = rows[:max_rows]

    h.update(np.int64(nrows).tobytes())
    h.update(np.int64(len(rows)).tobytes())

    for col in cols:
        if col not in tb.colnames():
            raise KeyError(f"Column '{col}' not in table {tabpath}")

        h.update(col.encode())

        data = tb.getcol(col)

        # CASA convention: last axis is row
        if isinstance(data, np.ndarray):
            data = np.take(data, rows, axis=data.ndim - 1)

            # Normalize dtype to avoid float/int ambiguity
            arr = np.ascontiguousarray(data)
            h.update(str(arr.dtype).encode())
            h.update(str(arr.shape).encode())
            h.update(arr.tobytes())
        else:
            # strings / lists
            for i in rows:
                h.update(str(data[i]).encode())

    tb.close()
    return h.hexdigest()

def copy_ms(src: str, dst: str):
    if not os.path.exists(src):
        raise RuntimeError(f"Missing input MS: {src}")
    if os.path.exists(dst):
        print(f"[WARN] Removing existing {dst}")
        shutil.rmtree(dst)
    print(f"[INFO] Copying MS: {src} -> {dst}")
    shutil.copytree(src, dst)


