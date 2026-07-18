from __future__ import annotations

from . import webserver_operational28 as base


VIDEO_FIRST_WIPE_CSS = """
    .live-wipe-video iframe { width:100%; height:100%; transform:none; }
    .live-wipe.zoom .live-wipe-video iframe { width:190%; height:190%; transform:translate(-23.7%,-23.7%); }
"""


VIDEO_FIRST_WIPE_JS = """$("liveWipeZoom").onclick = () => {
    box.classList.toggle("zoom");
    $("liveWipeZoom").textContent = box.classList.contains("zoom") ? "同意用" : "動画";
  };"""


HTML = (
    base.HTML
    .replace('<button id="liveWipeZoom" type="button">全体</button>', '<button id="liveWipeZoom" type="button">同意用</button>')
    .replace(
        '$("liveWipeZoom").onclick = () => { box.classList.toggle("zoom"); $("liveWipeZoom").textContent = box.classList.contains("zoom") ? "全体" : "拡大"; };',
        VIDEO_FIRST_WIPE_JS,
    )
    .replace("</style>", VIDEO_FIRST_WIPE_CSS + "\n  </style>")
)


def main(argv: list[str] | None = None) -> int:
    base.HTML = HTML
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

