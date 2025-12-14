import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from agents.UI_TARS.vision import save_and_encode, add_box_token
from agents.UI_TARS.agent_core import parse_action_block, execute_action, extract_thought, make_session_dir
from  agents.UI_TARS.llm_client import build_messages 
from evaluator.judge import judge_last_session_screenshot


@dataclass
class Transition:
    prompt_messages: List[dict] 
    response_text: str    
    screenshot_path: str
    done: bool


@dataclass
class Episode:
    session_dir: str
    transitions: List[Transition]
    evaluator_completed: int
    evaluator_justification: str


class DesktopRolloutEnv:
    """
    Real desktop rollouts:
      - screenshot
      - policy produces text containing Action(...)
      - execute action
      - repeat
      - evaluator judges final screenshot
    """
    def __init__(self, task: str, max_steps: int = 10, step_sleep: float = 1.0):
        self.task = task
        self.max_steps = max_steps
        self.step_sleep = step_sleep

    def run_episode(self, policy_generate_fn) -> Episode:
        """
        policy_generate_fn(messages)-> str
          Should return assistant text containing "Thought:" and "Action:".
        """
        history: List[dict] = []
        transitions: List[Transition] = []
        session_dir = make_session_dir()

        done = False
        for step in range(self.max_steps):
            screenshot_b64, screenshot_path = save_and_encode(step, session_dir)
            msgs = build_messages(self.task, history, screenshot_b64)

            raw_reply = policy_generate_fn(msgs)
            reply = add_box_token(raw_reply)

            action = parse_action_block(reply)
            status = execute_action(action)

            done = (status == "FINISHED")
            transitions.append(
                Transition(
                    prompt_messages=msgs,
                    response_text=reply,
                    screenshot_path=screenshot_path,
                    done=done
                )
            )

            history.append({"role": "assistant", "content": reply})

            if done:
                break

            time.sleep(self.step_sleep)

        verdict = judge_last_session_screenshot(self.task, session_dir)
        completed = int(verdict["completed"])
        justification = str(verdict["justification"])

        return Episode(
            session_dir=session_dir,
            transitions=transitions,
            evaluator_completed=completed,
            evaluator_justification=justification
        )
