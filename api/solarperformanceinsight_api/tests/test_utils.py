import datetime as dt
from functools import partial
from io import BytesIO, StringIO


from fastapi import HTTPException
import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import feather
import pytest


from solarperformanceinsight_api import utils


httpfail = partial(
    pytest.param, marks=pytest.mark.xfail(strict=True, raises=HTTPException)
)


@pytest.mark.parametrize(
    "inp,typ,exp",
    (
        (
            "time,datas\n2020-01-01T00:00Z,8.9",
            StringIO,
            pd.DataFrame({"time": [pd.Timestamp("2020-01-01T00:00Z")], "datas": [8.9]}),
        ),
        (
            b"time,datas\n2020-01-01T00:00Z,8.9",
            BytesIO,
            pd.DataFrame({"time": [pd.Timestamp("2020-01-01T00:00Z")], "datas": [8.9]}),
        ),
        (
            b"time,datas\n2020-01-01T00:00,8.9\n2020-01-02T00:00,-999",
            BytesIO,
            pd.DataFrame(
                {
                    "time": [
                        pd.Timestamp("2020-01-01T00:00"),
                        pd.Timestamp("2020-01-02T00:00"),
                    ],
                    "datas": [8.9, None],
                }
            ),
        ),
        # not valid later, but rely on dataframe validation to check dtypes
        (
            b"multi,header\ntime,datas\n2020-01-01T00:00,8.9\n2020-01-02T00:00,-999",
            BytesIO,
            pd.DataFrame(
                {
                    "multi": ["time", "2020-01-01T00:00", "2020-01-02T00:00"],
                    "header": ["datas", "8.9", np.nan],
                }
            ),
        ),
        # no header row
        httpfail(
            b"2020-01-01T00:00,8.9\n2020-01-02T00:00,-999",
            BytesIO,
            None,
        ),
        httpfail(
            "",
            StringIO,
            None,
        ),
        httpfail(
            "empty",
            StringIO,
            None,
        ),
        httpfail(
            "notenoughheaders,\na,b",
            StringIO,
            None,
        ),
        httpfail(
            "a,b\n0,1,2\n0,1,3,4,5,6",
            StringIO,
            None,
        ),
    ),
)
def test_read_csv(inp, typ, exp):
    out = utils.read_csv(typ(inp))
    pd.testing.assert_frame_equal(out, exp)


@pytest.mark.parametrize(
    "tbl,exp",
    (
        (
            pa.Table.from_arrays([[1.0, 2, 3], [4.0, 5, 6]], ["a", "b"]),
            pd.DataFrame({"a": [1, 2, 3.0], "b": [4, 5, 6.0]}),
        ),
        # complex types to test to_pandas
        (
            pa.Table.from_arrays(
                [pa.array([1.0, 2, 3]), pa.array([[], [5, 6], [7, 8]])], ["a", "b"]
            ),
            pd.DataFrame({"a": [1, 2, 3.0], "b": [[], [5, 6], [7, 8]]}),
        ),
        httpfail(
            b"notanarrowfile",
            None,
        ),
    ),
)
def test_read_arrow(tbl, exp):
    if isinstance(tbl, bytes):
        tblbytes = BytesIO(tbl)
    else:
        tblbytes = BytesIO(utils.dump_arrow_bytes(tbl))
    out = utils.read_arrow(tblbytes)
    pd.testing.assert_frame_equal(out, exp)


@pytest.mark.parametrize(
    "inp,exp",
    (
        ("text/csv", utils.read_csv),
        ("application/vnd.ms-excel", utils.read_csv),
        ("application/vnd.apache.arrow.file", utils.read_arrow),
        ("application/octet-stream", utils.read_arrow),
        httpfail("application/json", None),
    ),
)
def test_verify_content_type(inp, exp):
    out = utils.verify_content_type(inp)
    assert out == exp


@pytest.mark.parametrize(
    "inp,cols,exp",
    (
        (pd.DataFrame({"a": [0, 1], "b": [1, 2]}), ["a", "b"], set()),
        (
            pd.DataFrame(
                {"time": [pd.Timestamp("2020-01-01")], "b": [0.8], "c": ["notnumeric"]}
            ),
            ["time", "b"],
            {"c"},
        ),
        httpfail(
            pd.DataFrame({"time": [pd.Timestamp("2020-01-01")], "b": ["willfail"]}),
            ["time", "b"],
            set(),
        ),
        httpfail(pd.DataFrame({"a": [0, 1], "b": [1, 2]}), ["c"], {"a", "b"}),
        httpfail(pd.DataFrame({"time": [0, 1], "b": [1, 2]}), ["time", "b"], set()),
        httpfail(
            pd.DataFrame(
                {
                    "time": [
                        pd.Timestamp.now(),
                        pd.Timestamp("2020-01-01T00:00:01.09230"),
                    ],
                    "b": [1, 2],
                }
            ),
            ["time", "b"],
            set(),
        ),
    ),
)
def test_validate_dataframe(inp, cols, exp):
    out = utils.validate_dataframe(inp, cols)
    assert out == exp


@pytest.mark.parametrize(
    "df,tbl",
    (
        (
            pd.DataFrame({"a": [0.1, 0.2]}, dtype="float64"),
            pa.Table.from_arrays(
                [pa.array([0.1, 0.2], type=pa.float32())], names=["a"]
            ),
        ),
        (
            pd.DataFrame({"a": [0.1, 0.2]}, dtype="float32"),
            pa.Table.from_arrays(
                [pa.array([0.1, 0.2], type=pa.float32())], names=["a"]
            ),
        ),
        (
            pd.DataFrame(
                {
                    "a": [0.1, 0.2],
                    "time": [
                        pd.Timestamp("2020-01-01T00:00Z"),
                        pd.Timestamp("2020-01-02T00:00Z"),
                    ],
                },
            ),
            pa.Table.from_arrays(
                [
                    pa.array(
                        [
                            dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc),
                            dt.datetime(2020, 1, 2, tzinfo=dt.timezone.utc),
                        ],
                        type=pa.timestamp("s", tz="UTC"),
                    ),
                    pa.array([0.1, 0.2], type=pa.float32()),
                ],
                names=["time", "a"],
            ),
        ),
        (
            pd.DataFrame(
                {
                    "b": [-999, 129],
                    "time": [
                        pd.Timestamp("2020-01-01T00:00Z"),
                        pd.Timestamp("2020-01-02T00:00Z"),
                    ],
                    "a": [0.1, 0.2],
                },
            ),
            pa.Table.from_arrays(
                [
                    pa.array(
                        [
                            dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc),
                            dt.datetime(2020, 1, 2, tzinfo=dt.timezone.utc),
                        ],
                        type=pa.timestamp("s", tz="UTC"),
                    ),
                    pa.array([-999, 129], type=pa.float32()),
                    pa.array([0.1, 0.2], type=pa.float32()),
                ],
                names=["time", "b", "a"],
            ),
        ),
        httpfail(
            pd.DataFrame(
                {"a": [0.1, 0.2], "time": ["one", "two"]},
            ),
            None,
        ),
        httpfail(
            pd.DataFrame(
                {"a": [0.1, 0.2], "b": ["one", "two"]},
            ),
            None,
        ),
        # non-localized ok
        (
            pd.DataFrame(
                {
                    "b": [-999, 129],
                    "time": [
                        pd.Timestamp("2020-01-01T00:00"),
                        pd.Timestamp("2020-01-02T00:00"),
                    ],
                    "a": [0.1, 0.2],
                },
            ),
            pa.Table.from_arrays(
                [
                    pa.array(
                        [
                            dt.datetime(2020, 1, 1),
                            dt.datetime(2020, 1, 2),
                        ],
                        type=pa.timestamp("s"),
                    ),
                    pa.array([-999, 129], type=pa.float32()),
                    pa.array([0.1, 0.2], type=pa.float32()),
                ],
                names=["time", "b", "a"],
            ),
        ),
    ),
)
def test_convert_to_arrow(df, tbl):
    out = utils.convert_to_arrow(df)
    assert out == tbl


@pytest.mark.parametrize(
    "df",
    (
        pd.DataFrame(),
        pd.DataFrame({"a": [0, 1992.9]}),
        pd.DataFrame(
            {
                "b": [-999, 129],
                "time": [
                    pd.Timestamp("2020-01-01T00:00"),
                    pd.Timestamp("2020-01-02T00:00"),
                ],
                "a": [0.1, 0.2],
            },
        ),
        pd.DataFrame(
            {
                "b": [-999, 129],
                "time": [
                    pd.Timestamp("2020-01-01T00:00Z"),
                    pd.Timestamp("2020-01-02T00:00Z"),
                ],
                "a": [0.1, 0.2],
            },
        ),
    ),
)
def test_dump_arrow_bytes(df):
    tbl = pa.Table.from_pandas(df)
    out = utils.dump_arrow_bytes(tbl)
    assert isinstance(out, bytes)
    new = feather.read_feather(BytesIO(out))
    pd.testing.assert_frame_equal(df, new)
