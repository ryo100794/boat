from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .config import Settings
from .models import Ticket, VoteRequest


class VoteExecutionError(RuntimeError):
    pass


class VoteExecutor(Protocol):
    def execute(self, request: VoteRequest) -> list[dict[str, object]]: ...


@dataclass
class SeleniumVoteExecutor:
    settings: Settings

    LOGIN_MEMBER_XPATH = (
        '//*[@id="pwtautLoginDiv"]/section/div[1]/div/div/div[2]/div/input'
    )
    LOGIN_PIN_XPATH = (
        '//*[@id="pwtautLoginDiv"]/section/div[1]/div/div/div[3]/div/input'
    )
    LOGIN_MOBILE_XPATH = (
        '//*[@id="pwtautLoginDiv"]/section/div[1]/div/div/div[4]/div/input'
    )
    LOGIN_BUTTON_XPATH = (
        '//*[@id="pwtautLoginDiv"]/section/div[1]/div/div/div[6]/div/div/input'
    )
    SIMPLE_VOTE_XPATH = "/html/body/div[1]/section/div[2]/div/div[1]/ul/li[1]"
    STADIUM_XPATH = '//*[@id="one"]/div/form/div[1]/div[1]/div/input'
    REVIEW_XPATH = '//*[@id="one"]/div/form/div[14]/div'
    VOTE_BUTTON_ID = "btn-vote"
    RETURN_LINK_XPATH = '//*[@id="footer-link1"]/a'
    USER_AGENT = (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 12_0 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.0 "
        "Mobile/15E148 Safari/604.1"
    )

    def execute(self, request: VoteRequest) -> list[dict[str, object]]:
        driver = self._new_driver()
        submitted: list[dict[str, object]] = []
        try:
            wait = self._wait(driver)
            driver.get(self.settings.base_url)
            self._send(wait, self.LOGIN_MEMBER_XPATH, self.settings.member_number)
            self._send(wait, self.LOGIN_PIN_XPATH, self.settings.pin)
            self._send(
                wait,
                self.LOGIN_MOBILE_XPATH,
                self.settings.authorization_number_of_mobile,
            )
            self._click(wait, self.LOGIN_BUTTON_XPATH)
            for batch_number, tickets in enumerate(
                request.batches(self.settings.batch_size),
                start=1,
            ):
                self._submit_batch(driver, wait, request, tickets)
                submitted.append(
                    {
                        "batch": batch_number,
                        "tickets": len(tickets),
                        "stake_yen": sum(ticket.stake_yen for ticket in tickets),
                        "status": "submitted",
                    }
                )
            return submitted
        except Exception as exc:
            raise VoteExecutionError("browser vote execution failed") from exc
        finally:
            driver.quit()

    def _submit_batch(self, driver, wait, request: VoteRequest, tickets: tuple[Ticket, ...]):
        self._click(wait, self.SIMPLE_VOTE_XPATH)
        self._send(wait, self.STADIUM_XPATH, request.stadium.formal_tel_code)
        for row_number, ticket in enumerate(tickets, start=2):
            xpath = f"//*[@id='one']/div/form/div[{row_number}]/div[1]/div/input"
            self._send(wait, xpath, ticket.simple_betting_code(request.race_number))
        self._click(wait, self.REVIEW_XPATH)
        amount_xpath = (
            "/html/body/div[1]/form/section/div[1]/div/div/div"
            f"[{len(tickets) + 2}]/div/div/div[1]/div/input"
        )
        self._send(wait, amount_xpath, str(sum(ticket.stake_yen for ticket in tickets)))
        self._click(wait, self.VOTE_BUTTON_ID, by_id=True)
        self._click(wait, self.RETURN_LINK_XPATH)

    @staticmethod
    def _wait(driver):
        from selenium.webdriver.support.ui import WebDriverWait

        return WebDriverWait(driver, 10)

    @staticmethod
    def _send(wait, locator: str, value: str | None) -> None:
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as conditions

        element = wait.until(conditions.element_to_be_clickable((By.XPATH, locator)))
        element.clear()
        element.send_keys(value or "")

    @staticmethod
    def _click(wait, locator: str, *, by_id: bool = False) -> None:
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as conditions

        by = By.ID if by_id else By.XPATH
        wait.until(conditions.element_to_be_clickable((by, locator))).click()

    def _new_driver(self):
        try:
            from selenium import webdriver
        except ImportError as exc:
            raise VoteExecutionError(
                "selenium is required for explicitly enabled live voting"
            ) from exc
        options = webdriver.ChromeOptions()
        for argument in (
            "--headless=new",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            f"--user-agent={self.USER_AGENT}",
        ):
            options.add_argument(argument)
        return webdriver.Chrome(options=options)
