"""HackerNews tech digest: front page in, tech-only summary plus podcast-style MP3 out to Gmail.

Runs on behalf of the triggering user (`on_behalf_of=True`) so the `gmail` connector resolves that
user's credentials; meant for manual one-off executions, never for schedules.
"""

import json
from datetime import timedelta

import mistralai.workflows as workflows
import pydantic
from mistralai.workflows import Depends
from mistralai.workflows.plugins.mistralai.connectors import (
    ToolCallClient,
    connector,
    uses_connectors,
)

gmail_connector = connector("gmail")

HN_API_BASE = "https://hacker-news.firebaseio.com/v0"
CHAT_MODEL = "mistral-large-latest"
TTS_MODEL = "voxtral-mini-tts-3-0"
MAX_ATTACHMENT_B64_CHARS = 4_000_000


class HNTechDigestInput(pydantic.BaseModel):
    recipient_email: str = "anton.krapfenbacher@gmail.com"
    max_stories: int = 10
    candidate_count: int = 30
    language: str = "de"
    voice_id: str = "voice_481"


class Story(pydantic.BaseModel):
    id: int
    title: str
    url: str
    domain: str
    by: str
    score: int
    comments: int


class KeptStories(pydantic.BaseModel):
    kept_indices: list[int]


class StorySummary(pydantic.BaseModel):
    index: int
    title: str
    url: str
    summary: str


class DigestContent(pydantic.BaseModel):
    email_subject: str
    email_body: str
    audio_script: str
    stories: list[StorySummary]


class EmailSendResult(pydantic.BaseModel):
    message_id: str | None = None
    thread_id: str | None = None
    audio_bytes: int


class HNTechDigestOutput(pydantic.BaseModel):
    fetched_candidates: int
    tech_stories_kept: int
    email_subject: str
    email_body: str
    audio_script: str
    audio_filename: str
    audio_mp3_bytes: int
    email_message_id: str | None
    email_thread_id: str | None


def _language_name(code: str) -> str:
    return {"de": "German", "en": "English"}.get(code, code)


@workflows.activity(start_to_close_timeout=timedelta(seconds=90))
async def fetch_frontpage_stories(candidate_count: int) -> list[Story]:
    import asyncio

    import httpx

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{HN_API_BASE}/frontpage.json")
        response.raise_for_status()
        ids: list[int] = response.json()[:candidate_count]

        semaphore = asyncio.Semaphore(10)

        async def fetch_item(item_id: int) -> dict:
            async with semaphore:
                item_response = await client.get(f"{HN_API_BASE}/item/{item_id}.json")
                item_response.raise_for_status()
                return item_response.json()

        items = await asyncio.gather(*(fetch_item(item_id) for item_id in ids))

    stories: list[Story] = []
    for item in items:
        url = item.get("url")
        if (
            item.get("type") != "story"
            or not url
            or item.get("deleted")
            or item.get("dead")
        ):
            continue
        stories.append(
            Story(
                id=item["id"],
                title=item.get("title", ""),
                url=url,
                domain=url.split("//", 1)[-1].split("/", 1)[0],
                by=item.get("by", ""),
                score=item.get("score", 0),
                comments=item.get("descendants", 0),
            )
        )
    return stories


async def _chat_json(messages: list[dict], model: str = CHAT_MODEL) -> dict:
    from mistralai.workflows.plugins.mistralai import get_mistral_client

    client = get_mistral_client()
    response = await client.chat.complete_async(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    return json.loads(content)


@workflows.activity(start_to_close_timeout=timedelta(seconds=120))
async def select_tech_stories(stories: list[Story], max_stories: int) -> list[Story]:
    candidates = [
        {
            "index": position,
            "title": story.title,
            "domain": story.domain,
            "url": story.url,
        }
        for position, story in enumerate(stories, start=1)
    ]
    prompt = (
        "You are filtering the Hacker News front page for a tech-industry news digest.\n"
        "Keep only stories that are clearly about the technology industry: software, hardware, "
        "AI/ML, programming, cybersecurity, open source, developer tools, cloud, semiconductors, "
        "tech products and tech companies.\n"
        "Drop everything else: personal blogs and essays, Ask HN / Tell HN / Show HN self-posts, "
        "job postings, polls, politics, non-tech news, and anything without a real external article.\n"
        f"Keep at most {max_stories} stories, preserving front-page order (lowest index = most important).\n"
        'Respond with JSON only: {"kept_indices": [list of 1-based indices of kept stories]}. '
        f"Candidates:\n{json.dumps(candidates)}"
    )
    selection = KeptStories.model_validate(
        await _chat_json(
            [
                {
                    "role": "system",
                    "content": "You are a strict tech-news filter. Answer with JSON only.",
                },
                {"role": "user", "content": prompt},
            ]
        )
    )
    kept: list[Story] = []
    for index in selection.kept_indices:
        if 1 <= index <= len(stories) and stories[index - 1] not in kept:
            kept.append(stories[index - 1])
        if len(kept) >= max_stories:
            break
    return kept or stories[:max_stories]


@workflows.activity(start_to_close_timeout=timedelta(seconds=180))
async def compose_digest(stories: list[Story], language: str) -> DigestContent:
    language_name = _language_name(language)
    story_lines = "\n".join(
        f"{position}. {story.title} ({story.domain}) - {story.url}"
        for position, story in enumerate(stories, start=1)
    )
    prompt = (
        f"Write a Hacker News tech digest in {language_name} for these stories:\n{story_lines}\n\n"
        "Respond with JSON only, with this schema:\n"
        "{\n"
        '  "email_subject": string (short subject line),\n'
        '  "email_body": string (plain-text email: brief intro, then one numbered section per story '
        "with its headline, a 1-2 sentence summary and the link),\n"
        '  "audio_script": string (a podcast-style spoken intro of the day\'s top tech headlines: '
        "short greeting, one short paragraph per headline (say the headline, then the gist), brief outro. "
        "Target 150-220 words so the audio lands at roughly 1-2 minutes),\n"
        '  "stories": [{"index": 1-based position, "title": string, "url": string, '
        '"summary": string (1-2 sentences)}]\n'
        "}\n"
        f"Write everything in {language_name}."
    )
    content = DigestContent.model_validate(
        await _chat_json(
            [
                {
                    "role": "system",
                    "content": "You are a tech-news editor producing a concise digest and a podcast script. Answer with JSON only.",
                },
                {"role": "user", "content": prompt},
            ]
        )
    )
    return content


@workflows.activity(
    start_to_close_timeout=timedelta(minutes=4),
    retry_policy_max_attempts=1,
)
async def render_audio_and_send_email(
    recipient_email: str,
    subject: str,
    body: str,
    audio_script: str,
    audio_filename: str,
    voice_id: str,
    gmail: ToolCallClient = Depends(gmail_connector),
) -> EmailSendResult:
    from mistralai.workflows.plugins.mistralai import get_mistral_client

    client = get_mistral_client()
    speech = await client.audio.speech.complete_async(
        model=TTS_MODEL,
        input=audio_script,
        voice_id=voice_id,
        response_format="mp3",
    )
    audio_b64: str = speech.audio_data
    if len(audio_b64) > MAX_ATTACHMENT_B64_CHARS:
        raise workflows.WorkflowError(
            f"Rendered audio is too large to attach ({len(audio_b64)} base64 chars); shorten the script."
        )
    audio_bytes = len(audio_b64) * 3 // 4

    result = await gmail.call_tool(
        tool_name="send_email",
        arguments={
            "to": [recipient_email],
            "subject": subject,
            "body": body,
            "attachments": [
                {
                    "filename": audio_filename,
                    "mimeType": "audio/mpeg",
                    "content": audio_b64,
                }
            ],
        },
    )
    payload = (
        result
        if isinstance(result, dict)
        else getattr(result, "model_dump", lambda: {})()
    )
    return EmailSendResult(
        message_id=payload.get("id"),
        thread_id=payload.get("threadId"),
        audio_bytes=audio_bytes,
    )


@workflows.workflow.define(
    name="hackernews-tech-digest",
    workflow_display_name="HackerNews Tech Digest",
    workflow_description=(
        "Fetches the Hacker News front page, keeps tech-industry stories only, summarizes them, "
        "renders a 1-2 minute podcast-style MP3 and emails both to the triggering user's Gmail. "
        "Manual one-off runs only (on-behalf-of); no schedule."
    ),
    on_behalf_of=True,
)
@uses_connectors(gmail_connector)
class HackerNewsTechDigestWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, input: HNTechDigestInput) -> HNTechDigestOutput:
        stories = await fetch_frontpage_stories(input.candidate_count)
        tech_stories = await select_tech_stories(stories, input.max_stories)
        digest = await compose_digest(tech_stories, input.language)
        audio_filename = f"hn-tech-digest-{workflows.workflow.now():%Y%m%d-%H%M%S}.mp3"
        email = await render_audio_and_send_email(
            input.recipient_email,
            digest.email_subject,
            digest.email_body,
            digest.audio_script,
            audio_filename,
            input.voice_id,
        )
        return HNTechDigestOutput(
            fetched_candidates=len(stories),
            tech_stories_kept=len(tech_stories),
            email_subject=digest.email_subject,
            email_body=digest.email_body,
            audio_script=digest.audio_script,
            audio_filename=audio_filename,
            audio_mp3_bytes=email.audio_bytes,
            email_message_id=email.message_id,
            email_thread_id=email.thread_id,
        )
