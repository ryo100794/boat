from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from .config import Settings
from .login_probe import LoginProbeError, TeleboatLoginProbe
from .login_secrets import LoginSecrets
from .models import BetMethod, Ticket, VoteRequest


class VoteExecutionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        submission_may_have_occurred: bool = False,
        stage: str | None = None,
    ):
        super().__init__(message)
        self.submission_may_have_occurred = submission_may_have_occurred
        self.stage = stage


class VoteExecutor(Protocol):
    def execute(self, request: VoteRequest) -> list[dict[str, object]]: ...


@dataclass(frozen=True)
class ConfirmationSummary:
    tickets: int
    stake_yen: int
    unfinished: bool
    final_button_ready: bool


def verify_confirmation_text(
    text: str,
    *,
    request: VoteRequest,
    final_button_ready: bool,
) -> ConfirmationSummary:
    compact = re.sub(r"\s+", " ", text).strip()
    count_match = re.search(r"合計ベット数\s*([0-9,]+)ベット", compact)
    amount_match = re.search(r"購入金額\s*([0-9,]+)円", compact)
    if not count_match or not amount_match:
        raise VoteExecutionError("confirmation totals were not found")
    tickets = int(count_match.group(1).replace(",", ""))
    stake_yen = int(amount_match.group(1).replace(",", ""))
    if tickets != request.expanded_ticket_count:
        raise VoteExecutionError(
            f"confirmation ticket count mismatch: expected {request.expanded_ticket_count}, got {tickets}"
        )
    if stake_yen != request.total_stake_yen:
        raise VoteExecutionError(
            f"confirmation stake mismatch: expected {request.total_stake_yen}, got {stake_yen}"
        )
    if request.stadium.name not in compact or f"{request.race_number}R" not in compact:
        raise VoteExecutionError("confirmation race identity mismatch")
    if request.bet_type.label not in compact:
        raise VoteExecutionError("confirmation bet type mismatch")
    if request.method is BetMethod.REGULAR:
        for ticket in request.tickets:
            display = ticket.betting_number.display(request.bet_type)
            if display not in compact:
                raise VoteExecutionError(f"confirmation ticket missing: {display}")
    elif request.method is BetMethod.BOX:
        source = "".join(str(lane) for lane in request.source_positions[0])
        if "ボックス" not in compact or source not in compact:
            raise VoteExecutionError("confirmation box selection mismatch")
    else:
        source = "-".join(
            "".join(str(lane) for lane in position)
            for position in request.source_positions
        )
        if "フォーメーション" not in compact or source not in compact:
            raise VoteExecutionError("confirmation formation selection mismatch")
    unfinished = "本画面では投票未完了です" in compact
    if not unfinished:
        raise VoteExecutionError("official unfinished-wager marker was not found")
    if not final_button_ready:
        raise VoteExecutionError("final vote button was not ready")
    return ConfirmationSummary(
        tickets=tickets,
        stake_yen=stake_yen,
        unfinished=unfinished,
        final_button_ready=final_button_ready,
    )


@dataclass
class PlaywrightVoteExecutor:
    settings: Settings

    LOGIN_MEMBER_XPATH = '//input[@name="userId"]'
    LOGIN_PIN_XPATH = '//input[@name="pwd"]'
    LOGIN_MOBILE_XPATH = '//input[@name="pinNum"]'
    LOGIN_BUTTON_XPATH = '//input[@name="btnLogin"]'
    SIMPLE_VOTE_XPATH = '//a[normalize-space()="簡易投票する"]'
    STADIUM_XPATH = '//input[@name="jyoCode"]'
    REVIEW_XPATH = '//*[@id="btnAddList1"]'
    VOTE_BUTTON_ID = "btn-vote"
    RETURN_LINK_XPATH = '//*[@id="footer-link1"]/a'

    def preview(self, request: VoteRequest) -> dict[str, object]:
        return self._run(request, submit=False)

    def execute(self, request: VoteRequest) -> list[dict[str, object]]:
        return [self._run(request, submit=True)]

    def _run(self, request: VoteRequest, *, submit: bool) -> dict[str, object]:
        secrets = self._login_secrets()
        probe = TeleboatLoginProbe(timeout=30)
        authenticated = False
        final_triggered = False
        stage = "browser_start"
        result: dict[str, object] | None = None
        try:
            with probe._browser_page("mobile") as page:
                try:
                    stage = "login_form"
                    probe._open_official_page(page, "mobile")
                    if not probe._wait_for_login_form(page, "mobile"):
                        raise VoteExecutionError("official login form was not available")
                    stage = "authentication"
                    probe._submit_login_once(page, secrets)
                    authenticated = probe._wait_until_authenticated(page, "mobile")
                    if not authenticated:
                        raise VoteExecutionError("official login was not authenticated")
                    stage = "vote_menu"
                    self._open_vote_menu(page, probe, request)
                    stage = "ticket_input"
                    if request.method is BetMethod.REGULAR:
                        self._add_regular(page, request)
                    elif request.method is BetMethod.BOX:
                        self._add_box(page, request)
                    else:
                        self._add_formation(page, request)
                    stage = "confirmation"
                    summary = self._verify_confirmation(page, request)
                    result = {
                        "batch": 1,
                        "tickets": summary.tickets,
                        "stake_yen": summary.stake_yen,
                        "bet_type": request.bet_type.value,
                        "method": request.method.value,
                        "status": "preview_verified",
                        "final_button_clicked": False,
                        "verifications": {
                            "authentication": True,
                            "official_host_allowlist": True,
                            "mode_selection": True,
                            "ticket_inputs": True,
                            "official_expanded_count": (
                                True if request.method is not BetMethod.REGULAR else None
                            ),
                            "confirmation_identity": True,
                            "confirmation_selection": True,
                            "confirmation_ticket_count": True,
                            "confirmation_stake_yen": True,
                            "official_hidden_total": True,
                            "unfinished_marker": True,
                            "final_button_ready": True,
                            "final_amount_input": None,
                            "submission_completed": None,
                        },
                    }
                    if submit:
                        stage = "final_amount"
                        amount = self._visible(page, 'input[name="buyAmtSumInput"]')
                        amount.fill(str(request.total_stake_yen))
                        if amount.input_value() != str(request.total_stake_yen):
                            raise VoteExecutionError("final purchase amount verification failed")
                        result["verifications"]["final_amount_input"] = True
                        final_button = self._final_button(page)
                        stage = "final_submission"
                        final_triggered = True
                        try:
                            final_button.click()
                            page.wait_for_timeout(800)
                            completed = self._submission_completed(page)
                            result["status"] = (
                                "submitted_verified" if completed else "submission_unknown"
                            )
                            result["submission_verified"] = completed
                            result["final_button_clicked"] = True
                            result["verifications"]["submission_completed"] = completed
                        except Exception:
                            result["status"] = "submission_unknown"
                            result["submission_verified"] = False
                            result["final_button_clicked"] = True
                            result["verifications"]["submission_completed"] = False
                finally:
                    logout_confirmed = (
                        probe._logout(page, "mobile") if authenticated else False
                    )
                    if result is not None:
                        result["logout_confirmed"] = logout_confirmed
        except VoteExecutionError as exc:
            if exc.stage is None:
                exc.stage = stage
            raise
        except LoginProbeError as exc:
            raise VoteExecutionError(
                "browser vote execution failed",
                submission_may_have_occurred=final_triggered,
                stage=stage,
            ) from exc
        except Exception as exc:
            raise VoteExecutionError(
                "browser vote execution failed",
                submission_may_have_occurred=final_triggered,
                stage=stage,
            ) from exc
        if result is None:
            raise VoteExecutionError(
                "browser vote execution ended without a result",
                submission_may_have_occurred=final_triggered,
            )
        return result

    def _login_secrets(self) -> LoginSecrets:
        try:
            return LoginSecrets.parse(
                {
                    "mode": "mobile",
                    "member_number": self.settings.member_number,
                    "pin": self.settings.pin,
                    "auth_secret": self.settings.authorization_number_of_mobile,
                }
            )
        except Exception as exc:
            raise VoteExecutionError("live credentials are invalid or incomplete") from exc

    def _open_vote_menu(self, page, probe: TeleboatLoginProbe, request: VoteRequest) -> None:
        self._visible_from_locator(page.get_by_text(request.stadium.name, exact=True)).click()
        page.wait_for_timeout(400)
        self._visible_from_locator(page.get_by_text("投票する", exact=True)).click()
        page.wait_for_timeout(400)
        probe._assert_allowed_host(page.url, "mobile")

    def _select_mode(self, page, request: VoteRequest) -> None:
        decision = self._visible(page, 'input[type="submit"][value="決定"]')
        form = decision.locator("xpath=ancestor::form[1]")
        self._set_radio(form, "raceNo", str(request.race_number))
        self._set_radio(form, "syosiki", request.bet_type.official_value)
        self._set_radio(form, "betWay", request.method.official_value)
        decision.click()
        page.wait_for_timeout(800)

    def _add_regular(self, page, request: VoteRequest) -> None:
        batches = tuple(request.batches(10))
        for batch_index, batch in enumerate(batches):
            if batch_index:
                self._return_to_vote_selection(page)
            self._select_mode(page, request)
            form = page.locator('input[name="on1"]').locator("xpath=ancestor::form[1]")
            for index, ticket in enumerate(batch):
                form.locator(f'input[name="kumiTeiBanList[{index}]"]').fill(
                    ticket.betting_number.value
                )
                form.locator(f'input[name="buyAmtList[{index}]"]').fill(
                    str(ticket.quantity)
                )
                self._verify_input(
                    form.locator(f'input[name="kumiTeiBanList[{index}]"]'),
                    ticket.betting_number.value,
                    "combination",
                )
                self._verify_input(
                    form.locator(f'input[name="buyAmtList[{index}]"]'),
                    str(ticket.quantity),
                    "quantity",
                )
            form.locator('input[name="on1"]').click()
            self._wait_for_confirmation(page)

    def _add_box(self, page, request: VoteRequest) -> None:
        self._select_mode(page, request)
        submit = page.locator('input[name="forward_box"]')
        form = submit.locator("xpath=ancestor::form[1]")
        for lane in request.source_positions[0]:
            self._set_checkbox(form, "boxBetCheckList", lane - 1)
        form.locator("#betboxcal").click()
        page.wait_for_timeout(250)
        self._verify_calculated_count(page, request.expanded_ticket_count)
        amount = form.locator('input[name="buyAmt"]')
        amount.fill(str(request.quantity))
        self._verify_input(amount, str(request.quantity), "box quantity")
        submit.click()
        self._wait_for_confirmation(page)

    def _add_formation(self, page, request: VoteRequest) -> None:
        self._select_mode(page, request)
        submit = page.locator('input[name="forward_confirm"]')
        form = submit.locator("xpath=ancestor::form[1]")
        field_names = ("boxBetCheckList", "boatNoList2", "boatNoList3")
        for position_index, lanes in enumerate(request.source_positions):
            for lane in lanes:
                self._set_checkbox(form, field_names[position_index], lane - 1)
        form.locator("#calculate").click()
        page.wait_for_timeout(250)
        self._verify_calculated_count(page, request.expanded_ticket_count)
        amount = form.locator('input[name="buyAmt"]')
        amount.fill(str(request.quantity))
        self._verify_input(amount, str(request.quantity), "formation quantity")
        submit.click()
        self._wait_for_confirmation(page)

    def _return_to_vote_selection(self, page) -> None:
        self._visible(page, 'input[value="レース・投票方法を変更する"]').click()
        page.wait_for_timeout(400)
        self._visible_from_locator(page.get_by_text("投票する", exact=True)).click()
        page.wait_for_timeout(400)

    def _verify_confirmation(
        self,
        page,
        request: VoteRequest,
    ) -> ConfirmationSummary:
        final = self._final_button(page)
        summary = verify_confirmation_text(
            page.locator("body").inner_text(),
            request=request,
            final_button_ready=final.is_visible() and final.is_enabled(),
        )
        displayed_total = page.locator('input[name="buyAmtSumDisp"]')
        if displayed_total.count() != 1:
            raise VoteExecutionError("official hidden purchase total was not unique")
        self._verify_input(
            displayed_total,
            str(request.total_stake_yen),
            "official purchase total",
        )
        return summary

    @staticmethod
    def _wait_for_confirmation(page) -> None:
        page.get_by_text(
            "本画面では投票未完了です",
            exact=False,
        ).wait_for(state="visible")

    @staticmethod
    def _verify_calculated_count(page, expected: int) -> None:
        text = re.sub(r"\s+", " ", page.locator("body").inner_text())
        if f"{expected}ベット" not in text and f"ベット {expected}" not in text:
            raise VoteExecutionError(
                f"official expanded ticket count did not match {expected}"
            )

    @staticmethod
    def _verify_input(locator, expected: str, label: str) -> None:
        if locator.count() != 1 or locator.input_value() != expected:
            raise VoteExecutionError(f"{label} input verification failed")

    @staticmethod
    def _set_radio(form, name: str, value: str) -> None:
        candidates = form.locator(f'input[name="{name}"]')
        field = None
        for index in range(candidates.count()):
            candidate = candidates.nth(index)
            if str(candidate.get_attribute("value") or "").lstrip("0") == value.lstrip("0"):
                field = candidate
                break
        if field is None:
            raise VoteExecutionError(f"official radio was not found: {name}={value}")
        field.evaluate(
            "el => { el.checked=true; el.dispatchEvent(new Event('change',{bubbles:true})); }"
        )
        if not field.is_checked():
            raise VoteExecutionError(f"official radio verification failed: {name}={value}")

    @staticmethod
    def _set_checkbox(form, name: str, index: int) -> None:
        field = form.locator(f'input[name="{name}[{index}]"]')
        if field.count() != 1:
            raise VoteExecutionError(f"official checkbox was not found: {name}[{index}]")
        field.evaluate(
            "el => { el.checked=true; el.dispatchEvent(new Event('change',{bubbles:true})); }"
        )
        if not field.is_checked():
            raise VoteExecutionError(f"official checkbox verification failed: {name}[{index}]")

    def _final_button(self, page):
        by_id = page.locator(f"#{self.VOTE_BUTTON_ID}")
        if by_id.count() == 1:
            return by_id
        return self._visible(page, 'input[name="forward_bet"]')

    @staticmethod
    def _submission_completed(page) -> bool:
        text = re.sub(r"\s+", " ", page.locator("body").inner_text())
        return any(
            marker in text
            for marker in (
                "投票が完了しました",
                "投票を受け付けました",
                "投票受付結果",
            )
        )

    @classmethod
    def _visible(cls, page, selector: str):
        return cls._visible_from_locator(page.locator(selector), selector)

    @staticmethod
    def _visible_from_locator(candidates, selector: str = "locator"):
        for index in range(candidates.count() - 1, -1, -1):
            candidate = candidates.nth(index)
            if candidate.is_visible():
                return candidate
        raise VoteExecutionError(f"visible element was not found: {selector}")


SeleniumVoteExecutor = PlaywrightVoteExecutor
