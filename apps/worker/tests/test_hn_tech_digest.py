"""The hn-tech-digest workflow with every external effect mocked: HN API, LLM, TTS, Gmail."""

import asyncio
from datetime import timedelta

import mistralai.workflows as workflows
from mistralai.workflows.testing import create_test_worker
from worker.workflows import hn_tech_digest as mod

WORKFLOW_EXECUTION_TIMEOUT = timedelta(seconds=10)


@workflows.activity()
async def fetch_frontpage_stories(candidate_count: int) -> list[mod.Story]:
    return [
        mod.Story(
            id=1000 + i,
            title=f"Tech story {i}",
            url=f"https://tech.example.com/{i}",
            domain="tech.example.com",
            by="hnuser",
            score=100 - i,
            comments=5 * i,
        )
        for i in range(1, candidate_count + 1)
    ]


@workflows.activity()
async def select_tech_stories(
    stories: list[mod.Story], max_stories: int
) -> list[mod.Story]:
    return stories[:max_stories]


@workflows.activity()
async def compose_digest(stories: list[mod.Story], language: str) -> mod.DigestContent:
    return mod.DigestContent(
        email_subject="HN Tech Digest",
        email_body="digest body",
        audio_script="podcast script",
        stories=[
            mod.StorySummary(
                index=position, title=story.title, url=story.url, summary="summary"
            )
            for position, story in enumerate(stories, start=1)
        ],
    )


@workflows.activity()
async def render_audio_and_send_email(
    recipient_email: str,
    subject: str,
    body: str,
    audio_script: str,
    audio_filename: str,
    voice_id: str,
) -> mod.EmailSendResult:
    assert recipient_email == "anton.krapfenbacher@gmail.com"
    assert audio_filename.endswith(".mp3")
    return mod.EmailSendResult(message_id="msg-1", thread_id="thr-1", audio_bytes=12345)


async def test_hn_tech_digest_runs_end_to_end(temporal_env) -> None:
    async with create_test_worker(
        temporal_env,
        workflows=[mod.HackerNewsTechDigestWorkflow],
        activities=[
            fetch_frontpage_stories,
            select_tech_stories,
            compose_digest,
            render_audio_and_send_email,
        ],
    ):
        handle = await temporal_env.client.start_workflow(
            "hackernews-tech-digest",
            {
                "recipient_email": "anton.krapfenbacher@gmail.com",
                "max_stories": 3,
                "candidate_count": 5,
            },
            id="test-hn-tech-digest",
            task_queue="test-task-queue",
            execution_timeout=WORKFLOW_EXECUTION_TIMEOUT,
        )
        result = await asyncio.wait_for(handle.result(), timeout=15)
        assert result["fetched_candidates"] == 5
        assert result["tech_stories_kept"] == 3
        assert result["email_subject"] == "HN Tech Digest"
        assert result["email_message_id"] == "msg-1"
        assert result["email_thread_id"] == "thr-1"
        assert result["audio_mp3_bytes"] == 12345
        assert result["audio_filename"].startswith("hn-tech-digest-")
        assert result["audio_filename"].endswith(".mp3")
