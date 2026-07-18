from __future__ import annotations

from . import webserver_operational28 as base


COOKIE_SAFE_WIPE_CSS = """
    .live-wipe-video iframe { width:100%; height:100%; transform:none; }
    .live-wipe.zoom .live-wipe-video iframe { width:138%; height:138%; transform:translate(-13.8%,-13.8%); }
"""


HTML = (
    base.HTML
    .replace('class="live-wipe zoom hidden"', 'class="live-wipe hidden"')
    .replace('<button id="liveWipeZoom" type="button">全体</button>', '<button id="liveWipeZoom" type="button">拡大</button>')
    .replace("</style>", COOKIE_SAFE_WIPE_CSS + "\n  </style>")
)


def main(argv: list[str] | None = None) -> int:
    base.HTML = HTML
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

