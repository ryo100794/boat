from __future__ import annotations

from . import webserver_operational31 as base


CLOSE_WIPE_MARKUP = '<button id="liveWipeClose" type="button" title="このレースのワイプを閉じる">X</button>'


CLOSE_WIPE_JS = """const closedId = localStorage.getItem("boatLiveWipeClosedRaceId");
  if(closedId === String(r.race_id || "")){ box.classList.add("hidden"); return; }
  box.classList.remove("hidden");"""


CLOSE_WIPE_HANDLER_JS = """$("liveWipeClose").onclick = () => {
    localStorage.setItem("boatLiveWipeClosedRaceId", String(r.race_id || ""));
    box.classList.add("hidden");
  };
  $("liveWipeZoom").onclick = () => {
    box.classList.toggle("zoom");
    $("liveWipeZoom").textContent = box.classList.contains("zoom") ? "同意用" : "動画";
  };"""


HTML = (
    base.HTML
    .replace(
        '<button id="liveWipeZoom" type="button">動画</button>',
        CLOSE_WIPE_MARKUP + '\n          <button id="liveWipeZoom" type="button">動画</button>',
    )
    .replace('box.classList.remove("hidden");', CLOSE_WIPE_JS, 1)
    .replace(base.COOKIE_FIRST_WIPE_JS, CLOSE_WIPE_HANDLER_JS)
)


def main(argv: list[str] | None = None) -> int:
    base.HTML = HTML
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
