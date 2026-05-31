# SPDX-License-Identifier: Apache-2.0
"""Small valid audio payloads for voice upload contract checks."""

from __future__ import annotations

import base64
import gzip
from functools import lru_cache

VOICE_UPLOAD_FIXTURE_SIZES = {
    "mp3": 452,
    "flac": 8441,
    "ogg": 833,
    "aac": 941,
    "webm": 1179,
    "mp4": 1725,
}

_VOICE_UPLOAD_FIXTURE_GZIP_B64 = {
    "mp3": (
        "H4sIAAAAAAACE/N0MWZhAAEuBjj4/9jiAJzjmZeWD6T4gZiZgYFxA8MqQuAqIfCfEADZ65NYlqaQk5ibyoAEVJg+"
        "QBiMG3Y16bswkAb+P5Y4wsDjMWlKpT8DAwunBtP3ve+r731ZqukVNGXvu79/v78XWvnpnu6jPL3Us2HSAkyMXQe/"
        "//8XfP/iPxbL81+0Tmtl7ajP1Lp0E2QOJ5/FrTWMMwwYfkX6ZlxVt257arTi8lZN1osGa4w+yE1fkvnQySuhk1M5"
        "NOyC3xcJ7ag+oZbGNYsXrPFmatjulvbp/82+hgaQOdzcG+a5RB7mYJjV0KlxwMH7WK6Dg69LD1+ne9M6nUq7qJD3"
        "2qAQObL/5f//S30cfV2N9QwNDEJRAADw+K+GxAEAAA=="
    ),
    "flac": (
        "H4sIAAAAAAACE+3MvUsCcRwG8O8dZ3QgZiZUm3f8As0l7JamLqWwuKmxKaFOHEqHaK44CB3s8oIKgig7pJchqEVw"
        "MGnwhUrCISKHCJobGtrq/o3g+fAMz/Dw6FoiRkQyr/IqUdkJ901fRJz59iGUsgud9Vbjcv9dvx0ViDw9zlTXVzLL"
        "SafQJgUIAAAAAAAAAAAAAAAAAAAA/r/fn7Ve4nZYlJNdk58VpSrIG0QZoViM6Bfl5qudGzrJBU9z8S3NVphhNl5S"
        "wWFR2uvc8L6lh4rFxxNKNiSNsafriciAuHpkmmzbbp+5GauPhA4Di8lwWzKsPm9+zhejx1TUcD5sT8HPCoPnOmfU"
        "rDQrqDX/fTOeEMflefFAs8IlxkTpuSsyyWjeub3TQv+Uf3fW5eset+r5meoVp9bTf5UBI875IAAA"
    ),
    "ogg": (
        "H4sIAAAAAAACE/NPTw9mYGJAAyk7j2QxCvsXlBZ7pCamMDJaMDrIgyX8QcpRACMQe07dF8KoB1IekphezAYUSUvL"
        "LUhNB8mJAHFqXnJ+SmqRrU9iWbJCTmZSPlAl2CSWGVwIk0CumLbqlRjrf7Hu/5wzijzdNQ1WPVPYu+DdAadH/RIH"
        "HlW676x5+vTPcS7jeXzrJ+U4LPvko2LzP1v2i9AehvJrZwp1H4ba+zZ97m3Q9DixRCiQwUw83VDu2P8J+oZ9c6I8"
        "fs3crrFuY8m7bbN6beIyH8YpP6pbdlMvUPONThk358M5j7eb/Z9a53b+1GxxK17X4OX3J+6eNlG7POVqkvuXtCvv"
        "1/Zu+bKrgYHGgIkt0GCJSOyMst0bPv7b7vv/4OwbU0Kd5y5fofNSqeGs0MoZsmreKQs71j07sWP93tKjaqwcr9l2"
        "e0nfu2TWdiVT6bupi/SfsION/bxZM57r7hb5d7/Me7Fwy78rvxv6nRYvDIqd/KmnNWhZqYrngv1TD9d8mqW1K7Xx"
        "SGHOyn2rHh5fne57oaur+6WxwJq1XD/6dU4djJVRW6b53eCPcc2rrCpF51/hl3bv0EoSuMtq9fXbu4YZFen7hY57"
        "ee36dInv18SAj7PPH7u/uvTM56Mx39c+ajg5/6gku7FxStjy217pCq+/zFxwP8pyqnSvHn/d7I4VB5znzdWqTGXc"
        "fWV6Yq2grRLXsjmMi55pSIWzvLC+xXBhSdi8otufc/Tu38svt1QVXrWlsWjPMe1jeqv3HLvPfeRQoVnCu/vNGWlr"
        "Z73+pLx5d+PW8tV+2dnaQe8SaBElzD65ouY99+Tv5K1/H/RV2NVW1Hrv9X9bVgU0HhZ1fRe/BgCKXR3DQQMAAA=="
    ),
    "aac": (
        "H4sIAAAAAAACEwGtA1L8//FsQEhf/AEeNC4QdkkewpzdJTfE56iu+J3x1zM6q6rXNJE//oGlMcxbMu7/f62vHnro"
        "DM2YX1yHbXLGUeaUzbGWG5v9/eUNuLNmRsJpLSu77hzF7z/t/JlLmHnP/7zlmHtKXAccBMofDqhLM6pRSSWCgXVt"
        "KIYJ5O7kiUmBKkKuj0JG/vZNFK5+OilofVPu0qC6K6h01Rt7wmHX6SuAUtiub9K7vjGntMeE1LDrBznXtJR7R14Q"
        "qQY/kmk6ONoPF0GqnPhvhSiQPPya9GNEwViU2vw15LVZgeOEtZzKIUSXDHCURBEFRz8NfelZmDij27CmlvlZ57dg"
        "tpvqSvYVAnKYwU4NKelK+q3ZLT0vqnt2FNLfKzz2zi3S/TPb1FRLf0OTK2cWlOVHyKdTlMYKcEU5QecpxOU5YLqK"
        "iU8wHKJwbNbuyr2u7uwe10lQu7uwe10nYLikQy6SQXXSIFyewe10kguxkFN0nzG26S1lyfIbbpLRtTIL7k/MbRIa"
        "/anyF7EmWQ/MD+38h+rf6eEPLnwjJ+IPzCE/1VfksJ/kT/VKT/O97gIeyfBJD5GfnuIfrk/ISQ/Bj+Y8h+Q3mxP6"
        "efM8n9bf0YE/2S/kVJ/U78BxL7HetyH4jPvSQ+zv5wyH6uvxIEPgd8nyPyX+L5P8pf4uCf2a/IuT/Sn+IIn5U62S"
        "7V9LCH51fyrEPtx+G4h+Yr7mk+O+lSfgr63k/zz/mGJ/Zz7mE/x1fUYh7L+fZD4WfZch+er825D6+/P4h+Bn5U//"
        "8WxALX/8ARg0KLZqMziSAAAJJvjcnDciEsJsVtViLruc5S+prNtVilyOc7TWRLNtWVQzzx3quNMZLq2LQ5u592Li"
        "qpYXG3dWFY3mH7j3jHu6v9WPgelXHnUpFTCLUVoslStkqIiLZpGbCrEvkxIQyQgelO/t3/9/o2nDerfBbcbyPVeM"
        "vqbFbVYnLzibCpdSU9HFJZGdUgMmzYoVi1UnDNlRRYeTAaccZirwwFscZkrwwc6cZkfDBzbGZCrwcGxxYq8JBbHF"
        "irwkFscWB3cTZmB3wc6cZkfDBwZmQndwZmQndwNmYHdxNmYHdxNmZHwwc6cZkfDBwbGZCrwcWxxYq8JBb4b99GT4"
        "d9y+G8/u/8Qe8Pxmz+7/wb30ZPh33m+G/fQ/8J9y+M2f3f+IPeHJmd95vhv30ZPh33L4bz6H/iD3h+Mzvuf4N7w5"
        "MzvvN8N++h/4T7l8N59D/xB7w5Mzvuf4bz6H/ge8OTgDlDTerQMAAA=="
    ),
    "webm": (
        "H4sIAAAAAAACE5Nyvb94vlNbI6PTdyD+1Mji9LmRw6mppTw1KdepHchtbWSSCG5IZ2QAAZZsQd/ZJbt8d3cHr24R"
        "9VyZFrymcSGEJxayLhvIO+a7uwfIEwo5nB68ponRAMKVCd4NlGxiCX0DMYchkoFWAOSqBVrXNzbzOzn4NrT4JJal"
        "hTuCKZfODgdfiCKQY5+ug7ol5nojY/HRDqh+xjmNDEpb5zSX5qV0NDK0tTnG+weEBoetak7WWxC2u4XlyBaG5kam"
        "hxPnNzJu7XDY7wDWlJTSKJC8aLJ/QWmxR2piCiOjBaODPFgGFBAXi4vPJR/oTkbYkX5ipuvidlc/Z38X1yCX9h6g"
        "85IVcjKT8oEGpJ9Y6Lq4wyU0yDHE09/PpX2ygYEVGOkZmFoYgAGDvPO2UqeTzxsZFjtKNjIwNMwo8nTXNFj1TGHv"
        "gncHnB71Sxx4VOm+s+bp0z/HuYzn8a2flOOw7JOPis3/bNkvQnsYyq+dKdR9GGrv2/S5t0HT48QSoUAGM/F0Q7lj"
        "/yfoG/bNifL4NXO7xrqNJe+2zeq1ict8GKf8qG7ZTb1AzTc6ZdycD+c83m72f2qd2/lTs8WteF2Dl9+fuHvaRO3y"
        "lKtJ7l/Srrxf27vly64GBhoDJrZAgyUisTPKdm/4+G+77/+Ds29MCXWeu3yFzkulhrNCKxc79DcyiDbMkFXzTlnY"
        "se7ZiR3r95YeVWPleM2220v63iWztiuZSt9NXaT/hB1s7OfNmvFcd7fIv/tl3ouFW/5d+d3Q77R4YVDs5E89rUHL"
        "SlU8F+yferjm0yytXamNRwpzVu5b9fD46nTfC11d3S+NBdas5frRr3PqYKyM2jLN7wZ/jGteZVUpOv8Kv7R7h1aS"
        "wF1Wq6/f3jUscBRd6MjTyKDJMKMifb/QcS+vXZ8u8f2aGPBx9vlj91eXnvl8NOb72kcNJ+cflWQ3Nk4JW37bK13h"
        "9ZeZC+5HWU6V7tXjr5vdseKA87y5WpWpjLuvTE+sFbRV4lo2h3HRMw2pcJYX1rcYLiwJm1d0+3OO3v17+eWWqsKr"
        "tjQW7TmmfUxv9Z5j97mPHCo0S3h3vzkjbe2s15+UN+9u3Fq+2i87WzvoXQIt4ojZJ1fUvOee/J289e+Dvgq72opa"
        "773+b8uqgMbDoq7v4teULmo2TX8AKh4m7u7f3MiwvQtYDH1sYmz70MgMAEpIEiybBAAA"
    ),
    "mp4": (
        "H4sIAAAAAAACE2NgYJBJK6ksyCzOz2VgYGIA0UBslFtgYsjAwMCRVpSaysDAvDw3JbGEUc5ET6DMU+7QnLuq5kee"
        "r1j3Y+7H68ZWq1ddN5lo/69xqeGZaKN3/+vXrperesFzdkZ8THtu0bHAZ1PObpwmPfvv36e8OzanuR3K1NXe/U7m"
        "6Hv7t39mes+oPP9/z9MZ1V4x7DIsp+T5Vngbrwr0VG1qLM3VaON88u5Jp2ejltO6fie3f998RdbVWWlm1Aa/u7Rg"
        "l/aKkqvS1YcSr7/Ubgi6sS7/0u59hsu3HG+5suE1u+X1LVOq3eMEVrLZT8q0srjFL+64as6P/FaNCTZ/Zn1JdjkY"
        "MeXWH9MnWyMbH7dsnXNK0WU6T8EUF0FWd3ve2peRMywW396wbNrPyOfbE7bNfuX1TZSpaMZBP17Nl16/1t7UtdVf"
        "VV0mcum+ts23c7qX/hrfvhLiXe88WTtdbMpT9xPLg6cc4ypwtXR8rnnkqWXCrq5OfwOZRQU5196d2rvu3Ru5654B"
        "u3dvqI5VT9ixxFlvkmPpJYWYeQfflkzi3jghuGS+4baXuqknP0nnTbq41Yj7nf+ZXCGpvys/iW9Ui+Q/w//2T/ur"
        "+y8f8uvVKKs/4j+jaH81/MmG+U/8v3r5f977jknu5AdP/onz5z2SX+d/wpP/YP8zm/Yn5rOFf1V+tpl/7f7FRvub"
        "+k9C5l95f6BQf+O97Yo/en5fcn7z/jlP+6vXHxWcH5Sf/GSv+k9/8p+l//Q4/8760z35r+Y/hc6fwWsnvY335qgr"
        "rdc6YnenTrqjLmnfs8mP901Vf7B+7ZP/Nv9nJNWn2z0T/lNa66b4Zv/8FDux2umKP1/9uT3h1+/PP9ofpD/1Z5Qw"
        "0diWZWwxiYGBU+3HnTnmSkKHcsKuJuntnvNUf+Wa26FdMX3FW665bM6NDDE+L/tqx2VJvbXdzrN3fk96tGqauHR5"
        "mGjvM/kd33uqd+2/2t/4MlyuVFPUoDswSmeqdsqKju60ibO5NupPFhI4ySE35f3b+//rF2certp+MPfYJ9vwnn3L"
        "juaGqetbzOaaHhR88ajqxLlBzGpnu0S7Q9V5bga6tk9mXC6TpvXhQLRMmteHg+fmpLkfZjc7lqL1oSCnMGk9J+vG"
        "o117VMSOi7GXC6cllEMVsKU5lRekpaiXMwPFCs3SQJIpNQbFyyUnfjh4cONMrj0y0TIRqw855rf9/ZLyo/zOPunz"
        "7/4fkfvwJw1IH9wLErsHkrP/ML/ox9n5d/932D48OfN+5T7p7xcnP7xf9CPfrv2fwvf2x8b7nv8w32NpbP3e/HDd"
        "L/kf/jtrzOc7/xeoPjwZJAVSxr7H0gKYfT/l5ueXAbNtTm5ZRgoDCmB+ASSMGBgZQAgBGFFVofMdGPACJiByLClK"
        "zAayY0qywXYyYzHNCFmMkXR7VVJTSoqBtExqTnEJkg6guSxQvYw7c1MyE4EMhdwUdL/LA41jnRB6BMzRzUjJKYLJ"
        "FOeX5iGrDAbyUzwS81JyUkFqGFNyM/PSgAyB4lwUQ1VSIOIyKUWpaUgO4iktylGAsBk1ikuScoDsuuKS4hQkNXnA"
        "sjURw9uMDAIwpwKBWWpxSjE42hoaGlSBkixAWtxBFCy5u4GBYdE+VqAIq2hH2FMGNiCLERgVDCJJJUXQ4IGoAQVH"
        "cQk46KDRBbKRBWoz4wSQD4CuS0ZyDSNCHViuCi26rUGhArKruCQ5H0mfDhBLFacXpIA4Rfk5OTBz/v8Hm5SUXsCA"
        "kEG2xbY0pQQUHqa5qSWwcFFEjiRgxBYlFhTkIEcURyYwJQAA7AJRe70GAAA="
    ),
}


@lru_cache(maxsize=None)
def get_voice_upload_fixture(upload_format: str) -> bytes:
    try:
        encoded = "".join(_VOICE_UPLOAD_FIXTURE_GZIP_B64[upload_format])
    except KeyError as exc:
        raise ValueError(
            f"unknown voice upload fixture format: {upload_format}"
        ) from exc
    return gzip.decompress(base64.b64decode(encoded))
