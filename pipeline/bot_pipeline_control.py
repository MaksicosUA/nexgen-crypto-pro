#!/usr/bin/env python3
# ===== bot_pipeline_control.py =====
import asyncio
import os
import signal
import sys
import time
from pathlib import Path
import shutil
import zipfile
from datetime import datetime

import psutil
import yaml
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import (
    Message, FSInputFile,
    InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
)
from dotenv import load_dotenv, find_dotenv
from typing import Optional, List
import json
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext

# -----------------------------
# ENVIRONMENT LOADING (.env)
# priority: DOTENV_PATH -> nearest .env from CWD -> .env next to the script
# -----------------------------
ROOT = Path(__file__).resolve().parent

env_path = os.getenv("DOTENV_PATH") or find_dotenv(usecwd=True)
if not env_path:
    cand = ROOT / ".env"
    env_path = str(cand) if cand.exists() else ""

load_dotenv(env_path if env_path else None, override=False, verbose=True)
print(f"[DEBUG] .env loaded from: {env_path or '(none)'}")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_ID = int(os.getenv("TELEGRAM_ADMIN_ID", "0"))

# -----------------------------
# CONFIG LOOKUP (pipeline_config.yaml)
# priority: PIPELINE_CONFIG -> ./ -> ../ -> script directory
# -----------------------------
def resolve_config_path() -> Path:
    p_env = os.getenv("PIPELINE_CONFIG")
    if p_env and Path(p_env).exists():
        return Path(p_env).resolve()

    for cand in [
        Path.cwd() / "pipeline_config.yaml",
        ROOT.parent / "pipeline_config.yaml",
        ROOT / "pipeline_config.yaml",
    ]:
        if cand.exists():
            return cand.resolve()

    raise FileNotFoundError(
        "pipeline_config.yaml not found (checked $PIPELINE_CONFIG, ./, ../ and the script directory)"
    )

CONFIG_PATH = resolve_config_path()
print(f"[DEBUG] using config: {CONFIG_PATH}")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

# -----------------------------
# PIPELINE CONFIG LOADING (base template, do NOT modify)
# -----------------------------
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f) or {}

CONFIG_DIR = CONFIG_PATH.parent

def resolve_config_path_value(value, default: Path) -> Path:
    if value in (None, ""):
        return default.resolve()
    path = Path(value)
    return path.resolve() if path.is_absolute() else (CONFIG_DIR / path).resolve()

LOG_FILE = resolve_config_path_value(cfg.get("log_file"), ROOT / "logs" / "crypto_log.txt")
PLOT_PATH = resolve_config_path_value(cfg.get("plot_path"), ROOT / "plot")

# -----------------------------
# PIPELINE SUBPROCESS PARAMETERS
# -----------------------------
PIPE_SCRIPT = cfg.get("pipeline_script", "crypto_pipeline_v4.3.py")
# Priority: PIPELINE_PYTHON (env) → pipeline_python (config) → python from the current environment
# IMPORTANT: use sys.executable without resolve(), otherwise the venv symlink resolves to the system Python
_pipe_python = (
    os.getenv("PIPELINE_PYTHON")
    or cfg.get("pipeline_python")
    or sys.executable
)
PIPE_CMD = [_pipe_python, str((ROOT / PIPE_SCRIPT).resolve())]

proc: psutil.Popen | None = None
started_at = 0.0
CALLBACK_RATE_LIMIT_SECONDS = 1.0
_last_callback_ts: dict[int, float] = {}

# -----------------------------
# BOT/DISPATCHER
# -----------------------------
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

# ACCESS STORAGE

ACCESS_FILE = (ROOT / "access.json").resolve()

def load_access():
    if ACCESS_FILE.exists():
        try:
            data = json.loads(ACCESS_FILE.read_text(encoding="utf-8"))
            data.setdefault("operators", [])
            data.setdefault("roles", {})
            data.setdefault("participants", {})
            data.setdefault("access_requests", {})
            return data
        except Exception:
            pass
    return {"operators": [], "roles": {}, "participants": {}, "access_requests": {}}

def save_access(data):
    ACCESS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

ACCESS = load_access()
USER_ACTIVITY_DIR = (ROOT / "logs" / "user_activity").resolve()
USER_ACTIVITY_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# ACCESS CONTROL
# -----------------------------

def is_admin_id(user_id: Optional[int]) -> bool:
    return user_id is not None and (
        user_id == ADMIN_ID
        or ACCESS.get("roles", {}).get(str(user_id)) == "admin"
    )


def is_primary_admin_id(user_id: Optional[int]) -> bool:
    return user_id is not None and user_id == ADMIN_ID


def now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def describe_user(user) -> dict:
    if not user:
        return {}
    full_name = " ".join(part for part in [user.first_name, user.last_name] if part).strip()
    return {
        "id": user.id,
        "username": user.username or "",
        "full_name": full_name,
        "language_code": getattr(user, "language_code", "") or "",
    }


def update_participant_record(user, action: str) -> None:
    info = describe_user(user)
    user_id = info.get("id")
    if not user_id:
        return
    participants = ACCESS.setdefault("participants", {})
    record = participants.get(str(user_id), {})
    if not record:
        record["first_seen"] = now_iso()
        record["interaction_count"] = 0
    record.update({
        "username": info.get("username", ""),
        "full_name": info.get("full_name", ""),
        "language_code": info.get("language_code", ""),
        "last_seen": now_iso(),
        "last_action": action,
        "role": get_role(user_id),
    })
    record["interaction_count"] = int(record.get("interaction_count", 0)) + 1
    participants[str(user_id)] = record
    save_access(ACCESS)


def append_user_activity(user, action: str) -> None:
    info = describe_user(user)
    user_id = info.get("id")
    if not user_id:
        return
    log_path = USER_ACTIVITY_DIR / f"{user_id}.log"
    username = f"@{info['username']}" if info.get("username") else "-"
    full_name = info.get("full_name") or "-"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{now_iso()}] {action} | username={username} | name={full_name}\n")


def track_user_action(user, action: str) -> None:
    update_participant_record(user, action)
    append_user_activity(user, action)


def recent_user_activity(user_id: int, lines: int = 15) -> str:
    log_path = USER_ACTIVITY_DIR / f"{user_id}.log"
    if not log_path.exists():
        return "(no actions yet)"
    content = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not content:
        return "(no actions yet)"
    return "\n".join(content[-lines:])


def request_access(user, note: str = "") -> tuple[bool, str]:
    info = describe_user(user)
    user_id = info.get("id")
    if not user_id:
        return False, "Could not identify the user."
    requests = ACCESS.setdefault("access_requests", {})
    existing = requests.get(str(user_id), {})
    if existing.get("status") == "pending":
        return False, "Your request has already been submitted and is awaiting admin review."
    requests[str(user_id)] = {
        "status": "pending",
        "submitted_at": now_iso(),
        "note": note,
        "username": info.get("username", ""),
        "full_name": info.get("full_name", ""),
    }
    save_access(ACCESS)
    return True, "✅ Access request sent to the administrator. Please wait for a decision."


def resolve_access_request(user_id: int, status: str) -> None:
    requests = ACCESS.setdefault("access_requests", {})
    req = requests.get(str(user_id))
    if not req:
        return
    req["status"] = status
    req["resolved_at"] = now_iso()
    requests[str(user_id)] = req


def format_participant_brief(user_id: str, data: dict) -> str:
    role = get_role(int(user_id))
    full_name = data.get("full_name") or "-"
    username = f"@{data.get('username')}" if data.get("username") else "-"
    return (
        f"{user_id} | role: {role} | {full_name} | {username} | "
        f"last seen: {data.get('last_seen', '-') }"
    )


def format_request_brief(user_id: str, data: dict) -> str:
    full_name = data.get("full_name") or "-"
    username = f"@{data.get('username')}" if data.get("username") else "-"
    return f"{user_id} | {full_name} | {username} | requested: {data.get('submitted_at', '-')}"


def participants_menu_kb(limit: int = 15) -> InlineKeyboardMarkup:
    participants = ACCESS.get("participants", {})
    items = sorted(
        participants.items(),
        key=lambda item: item[1].get("last_seen", ""),
        reverse=True,
    )[:limit]
    rows = []
    for uid, data in items:
        name = data.get("full_name") or data.get("username") or uid
        rows.append([
            InlineKeyboardButton(
                text=f"👤 {name} [{uid}]",
                callback_data=f"admin_userlog:{uid}",
            )
        ])
    rows.append([InlineKeyboardButton(text="⬅️ Back", callback_data="admin_back")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def active_participants_menu_kb(limit: int = 20) -> InlineKeyboardMarkup:
    roles = ACCESS.get("roles", {})
    participants = ACCESS.get("participants", {})
    items = []
    for uid, role in roles.items():
        if uid == str(ADMIN_ID):
            continue
        pdata = participants.get(uid, {})
        items.append((uid, role, pdata))
    items.sort(key=lambda item: item[2].get("last_seen", ""), reverse=True)

    rows = []
    for uid, role, pdata in items[:limit]:
        name = pdata.get("full_name") or pdata.get("username") or uid
        rows.append([
            InlineKeyboardButton(text=f"👁 {name} [{role}]", callback_data=f"admin_userlog:{uid}"),
            InlineKeyboardButton(text="🗑", callback_data=f"admin_remove_participant:{uid}"),
        ])

    rows.append([InlineKeyboardButton(text="➕ Add manually", callback_data="admin_grant")])
    rows.append([InlineKeyboardButton(text="⬅️ Back", callback_data="admin_back")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def access_requests_menu_kb(limit: int = 20) -> InlineKeyboardMarkup:
    requests = ACCESS.get("access_requests", {})
    items = [
        (uid, data)
        for uid, data in requests.items()
        if data.get("status") == "pending"
    ]
    items.sort(key=lambda item: item[1].get("submitted_at", ""), reverse=True)

    rows = []
    for uid, data in items[:limit]:
        name = data.get("full_name") or data.get("username") or uid
        rows.append([
            InlineKeyboardButton(text=f"👁 {name}", callback_data=f"admin_request_view:{uid}"),
            InlineKeyboardButton(text="➕", callback_data=f"admin_request_add:{uid}"),
            InlineKeyboardButton(text="🗑", callback_data=f"admin_request_delete:{uid}"),
        ])

    rows.append([InlineKeyboardButton(text="⬅️ Back", callback_data="admin_back")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def request_actions_kb(user_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="➕ Grant access", callback_data=f"admin_request_add:{user_id}")],
        [InlineKeyboardButton(text="🗑 Delete request", callback_data=f"admin_request_delete:{user_id}")],
        [InlineKeyboardButton(text="⬅️ Back to requests", callback_data="admin_requests")],
    ])


def participant_actions_kb(user_id: int) -> InlineKeyboardMarkup:
    if is_primary_admin_id(user_id):
        return InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🔒 Primary admin is protected", callback_data="locked:primary_admin")],
            [InlineKeyboardButton(text="⬅️ Back to participants", callback_data="admin_active_participants")],
        ])
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🗑 Remove from participants", callback_data=f"admin_remove_participant:{user_id}")],
        [InlineKeyboardButton(text="⬅️ Back to participants", callback_data="admin_active_participants")],
    ])


def role_selection_kb(prefix: str = "admin_set_role") -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="👁 viewer", callback_data=f"{prefix}:viewer")],
        [InlineKeyboardButton(text="🛠 operator", callback_data=f"{prefix}:operator")],
        [InlineKeyboardButton(text="🧠 model", callback_data=f"{prefix}:model")],
        [InlineKeyboardButton(text="👑 admin", callback_data=f"{prefix}:admin")],
        [InlineKeyboardButton(text="⬅️ Cancel", callback_data="admin_back")],
    ])


def request_role_selection_kb(user_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="👁 Grant viewer", callback_data=f"admin_request_add_role:{user_id}:viewer")],
        [InlineKeyboardButton(text="🛠 Grant operator", callback_data=f"admin_request_add_role:{user_id}:operator")],
        [InlineKeyboardButton(text="🧠 Grant model", callback_data=f"admin_request_add_role:{user_id}:model")],
        [InlineKeyboardButton(text="👑 Grant admin", callback_data=f"admin_request_add_role:{user_id}:admin")],
        [InlineKeyboardButton(text="⬅️ Back to requests", callback_data="admin_requests")],
    ])


def confirm_request_delete_kb(user_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="✅ Yes, delete the request", callback_data=f"admin_request_delete_confirm:{user_id}")],
        [InlineKeyboardButton(text="❌ Cancel", callback_data="admin_requests")],
    ])


def confirm_participant_delete_kb(user_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="✅ Yes, remove the participant", callback_data=f"admin_remove_participant_confirm:{user_id}")],
        [InlineKeyboardButton(text="❌ Cancel", callback_data="admin_active_participants")],
    ])


def guest_panel_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🆔 Show my ID", callback_data="guest_my_id")],
        [InlineKeyboardButton(text="📝 Request access", callback_data="guest_apply_access")],
    ])


async def notify_admin_about_request(user) -> None:
    if not ADMIN_ID:
        return
    info = describe_user(user)
    username = f"@{info['username']}" if info.get("username") else "-"
    full_name = info.get("full_name") or "-"
    text = (
        "📝 New access request\n"
        f"ID: {info.get('id')}\n"
        f"Name: {full_name}\n"
        f"Username: {username}\n"
        f"Time: {now_iso()}"
    )
    try:
        await bot.send_message(ADMIN_ID, text)
    except Exception:
        pass


ACCESS_ROLE_ALIASES = {
    "admin": "admin",
    "free": "viewer",
    "viewer": "viewer",
    "read": "viewer",
    "readonly": "viewer",
    "paid": "operator",
    "vip": "operator",
    "private": "operator",
    "operator": "operator",
    "model": "model",
}

ROLE_SET = {"viewer", "operator", "model", "admin"}
SESSION_ROLE_SET = {"operator", "model"}


def normalize_role(role: Optional[str]) -> str:
    if not role:
        return ""
    return ACCESS_ROLE_ALIASES.get(role.strip().lower(), "")


def sync_user_access(user_id: int, role: Optional[str]) -> None:
    normalized = normalize_role(role)
    roles = ACCESS.setdefault("roles", {})
    ops = set(ACCESS.get("operators", []))

    if is_primary_admin_id(user_id):
        ACCESS["operators"] = sorted(ops)
        return

    if normalized:
        roles[str(user_id)] = normalized
    else:
        roles.pop(str(user_id), None)

    if normalized in SESSION_ROLE_SET:
        ops.add(user_id)
    else:
        ops.discard(user_id)

    ACCESS["operators"] = sorted(ops)

async def ensure_admin_message(message: Message) -> bool:
    uid = message.from_user.id if message.from_user else None
    track_user_action(message.from_user, f"message:{message.text or '[non-text]'}")
    if not is_admin_id(uid):
        await message.answer("⛔ Access denied.")
        return False
    return True

async def ensure_admin_callback(call: CallbackQuery) -> bool:
    if not await enforce_callback_rate_limit(call):
        return False
    uid = call.from_user.id if call.from_user else None
    track_user_action(call.from_user, f"callback:{call.data}")
    if not is_admin_id(uid):
        await call.answer("⛔ Access denied.", show_alert=True)
        return False
    return True

def is_operator_id(user_id: Optional[int]) -> bool:
    if user_id is None:
        return False
    return (user_id == ADMIN_ID) or (user_id in ACCESS.get("operators", []))

def get_role(user_id: Optional[int]) -> str:
    if user_id is None:
        return "none"
    if user_id == ADMIN_ID:
        return "admin"
    stored_role = ACCESS.get("roles", {}).get(str(user_id))
    normalized = normalize_role(stored_role)
    if normalized:
        return normalized
    if user_id in ACCESS.get("operators", []):
        return "operator"
    return "none"


def has_panel_access(user_id: Optional[int]) -> bool:
    if is_admin_id(user_id):
        return True
    return get_role(user_id) in ROLE_SET


def has_session_access(user_id: Optional[int]) -> bool:
    if is_admin_id(user_id):
        return True
    return get_role(user_id) in SESSION_ROLE_SET


def has_model_access(user_id: Optional[int]) -> bool:
    if is_admin_id(user_id):
        return True
    return get_role(user_id) == "model"


async def ensure_panel_message(message: Message) -> bool:
    uid = message.from_user.id if message.from_user else None
    track_user_action(message.from_user, f"message:{message.text or '[non-text]'}")
    if not has_panel_access(uid):
        await message.answer(
            "⛔ Panel access has not been granted. Ask an administrator to assign you a role.\n"
            "To find out your Telegram ID, send /whoami or /id and forward the result to an administrator.",
            reply_markup=guest_panel_kb()
        )
        return False
    return True


async def ensure_panel_callback(call: CallbackQuery) -> bool:
    if not await enforce_callback_rate_limit(call):
        return False
    uid = call.from_user.id if call.from_user else None
    track_user_action(call.from_user, f"callback:{call.data}")
    if not has_panel_access(uid):
        await call.answer("⛔ Panel access has not been granted. Use /whoami or /id to find out your Telegram ID.", show_alert=True)
        return False
    return True

async def ensure_operator_message(message: Message) -> bool:
    uid = message.from_user.id if message.from_user else None
    track_user_action(message.from_user, f"message:{message.text or '[non-text]'}")
    if not has_session_access(uid):
        await message.answer("⛔ This command requires the operator or model role.")
        return False
    return True

async def ensure_operator_callback(call: CallbackQuery) -> bool:
    if not await enforce_callback_rate_limit(call):
        return False
    uid = call.from_user.id if call.from_user else None
    track_user_action(call.from_user, f"callback:{call.data}")
    if not has_session_access(uid):
        await call.answer("⛔ This feature requires the operator or model role.", show_alert=True)
        return False
    return True


async def ensure_model_message(message: Message) -> bool:
    uid = message.from_user.id if message.from_user else None
    track_user_action(message.from_user, f"message:{message.text or '[non-text]'}")
    if not has_model_access(uid):
        await message.answer("⛔ This command requires the model role.")
        return False
    return True


async def ensure_model_callback(call: CallbackQuery) -> bool:
    if not await enforce_callback_rate_limit(call):
        return False
    uid = call.from_user.id if call.from_user else None
    track_user_action(call.from_user, f"callback:{call.data}")
    if not has_model_access(uid):
        await call.answer("⛔ This feature requires the model role.", show_alert=True)
        return False
    return True


async def enforce_callback_rate_limit(call: CallbackQuery, min_interval: float = CALLBACK_RATE_LIMIT_SECONDS) -> bool:
    uid = call.from_user.id if call.from_user else None
    if uid is None:
        return True
    now = time.monotonic()
    last = _last_callback_ts.get(uid, 0.0)
    if now - last < min_interval:
        await call.answer("Too many requests. Please press buttons no more than once per second.", show_alert=False)
        return False
    _last_callback_ts[uid] = now
    return True

# Roles and checks

# FSM states for entering user_id/role
class AdminForm(StatesGroup):
    grant_user = State()
    deny_user = State()
    role_user = State()
    role_value = State()

class DaysForm(StatesGroup):
    custom_days = State()  # manual input of the number of days

# Admin keyboards

def admin_menu_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="➕ Grant access", callback_data="admin_grant")],
        [InlineKeyboardButton(text="🚫 Revoke access", callback_data="admin_deny")],
        [InlineKeyboardButton(text="🎭 Assign role", callback_data="admin_role")],
        [InlineKeyboardButton(text="📋 Operator list", callback_data="admin_list")],
        [InlineKeyboardButton(text="👑 Admin list", callback_data="admin_list_admins")],
        [InlineKeyboardButton(text="📝 Requests", callback_data="admin_requests")],
        [InlineKeyboardButton(text="✅ Active participants", callback_data="admin_active_participants")],
        [InlineKeyboardButton(text="👥 Participants", callback_data="admin_participants")],
        [InlineKeyboardButton(text="⬅️ Back", callback_data="back_panel")],
    ])

# Admin command to open the menu
@dp.message(Command("admin"))
async def cmd_admin(message: Message):
    if not await ensure_admin_message(message): return
    await message.answer("👑 Admin menu", reply_markup=admin_menu_kb())

# Operator list
@dp.callback_query(F.data == "admin_list")
async def cb_admin_list(call: CallbackQuery):
    if not await ensure_admin_callback(call): return
    roles = ACCESS.get("roles", {})
    lines = []
    if ADMIN_ID:
        lines.append(f"{ADMIN_ID}: admin (main)")
    for uid, role in sorted(roles.items(), key=lambda item: int(item[0])):
        if uid == str(ADMIN_ID):
            continue
        lines.append(f"{uid}: {normalize_role(role) or role}")
    if not lines:
        await call.message.answer("The access list is empty.")
        await call.answer()
        return
    body = "\n".join(lines)
    await call.message.answer("Assigned roles:\n" + body)
    await call.answer()


@dp.callback_query(F.data == "admin_list_admins")
async def cb_admin_list_admins(call: CallbackQuery):
    if not await ensure_admin_callback(call):
        return
    roles = ACCESS.get("roles", {})
    admin_ids = set()
    if ADMIN_ID:
        admin_ids.add(str(ADMIN_ID))
    admin_ids.update(uid for uid, role in roles.items() if normalize_role(role) == "admin")

    if not admin_ids:
        await call.message.answer("The administrator list is empty.")
        await call.answer()
        return

    lines = []
    participants = ACCESS.get("participants", {})
    for uid in sorted(admin_ids, key=lambda item: int(item)):
        pdata = participants.get(uid, {})
        name = pdata.get("full_name") or "-"
        username = f"@{pdata.get('username')}" if pdata.get("username") else "-"
        suffix = " (main)" if uid == str(ADMIN_ID) else ""
        lines.append(f"{uid}: admin{suffix} | {name} | {username}")

    await call.message.answer("👑 Administrators:\n" + "\n".join(lines))
    await call.answer()

@dp.callback_query(F.data == "admin_participants")
async def cb_admin_participants(call: CallbackQuery):
    if not await ensure_admin_callback(call): return
    participants = ACCESS.get("participants", {})
    if not participants:
        await call.message.answer("No participants have interacted with the bot yet.")
        await call.answer()
        return
    body = "\n".join(
        format_participant_brief(uid, data)
        for uid, data in sorted(
            participants.items(),
            key=lambda item: item[1].get("last_seen", ""),
            reverse=True,
        )[:10]
    )
    await call.message.answer(
        "👥 Recent participants:\n" + body + "\n\nTap a user below to view their activity log.",
        reply_markup=participants_menu_kb(),
    )
    await call.answer()


@dp.callback_query(F.data == "admin_active_participants")
async def cb_admin_active_participants(call: CallbackQuery):
    if not await ensure_admin_callback(call): return
    roles = {uid: role for uid, role in ACCESS.get("roles", {}).items() if uid != str(ADMIN_ID)}
    if not roles:
        await call.message.answer(
            "✅ There are currently no active participants with granted access.",
            reply_markup=active_participants_menu_kb(),
        )
        await call.answer()
        return

    participants = ACCESS.get("participants", {})
    body = "\n".join(
        format_participant_brief(uid, participants.get(uid, {}))
        for uid in sorted(roles.keys(), key=lambda item: participants.get(item, {}).get("last_seen", ""), reverse=True)[:15]
    )
    await call.message.answer(
        "✅ Active participants:\n" + body + "\n\nTap a user to view details or 🗑 to remove them from the list.",
        reply_markup=active_participants_menu_kb(),
    )
    await call.answer()


@dp.callback_query(F.data == "admin_requests")
async def cb_admin_requests(call: CallbackQuery):
    if not await ensure_admin_callback(call): return
    pending = [
        (uid, data)
        for uid, data in ACCESS.get("access_requests", {}).items()
        if data.get("status") == "pending"
    ]
    if not pending:
        await call.message.answer(
            "📝 There are currently no pending access requests.",
            reply_markup=access_requests_menu_kb(),
        )
        await call.answer()
        return

    body = "\n".join(
        format_request_brief(uid, data)
        for uid, data in sorted(pending, key=lambda item: item[1].get("submitted_at", ""), reverse=True)[:15]
    )
    await call.message.answer(
        "📝 Pending requests:\n" + body + "\n\nTap 👁 to view, ➕ to approve, 🗑 to delete a request.",
        reply_markup=access_requests_menu_kb(),
    )
    await call.answer()

@dp.callback_query(F.data == "admin_back")
async def cb_admin_back(call: CallbackQuery):
    if not await ensure_admin_callback(call): return
    await call.message.answer("👑 Admin menu", reply_markup=admin_menu_kb())
    await call.answer()


@dp.callback_query(F.data == "admin_open")
async def cb_admin_open(call: CallbackQuery):
    if not await ensure_admin_callback(call): return
    await call.message.answer("👑 Admin menu", reply_markup=admin_menu_kb())
    await call.answer()

@dp.callback_query(F.data.startswith("admin_userlog:"))
async def cb_admin_userlog(call: CallbackQuery):
    if not await ensure_admin_callback(call): return
    uid_text = call.data.split(":", 1)[1]
    if not uid_text.isdigit():
        await call.answer("Invalid user_id", show_alert=True)
        return
    uid = int(uid_text)
    pdata = ACCESS.get("participants", {}).get(str(uid), {})
    username = f"@{pdata.get('username')}" if pdata.get("username") else "-"
    full_name = pdata.get("full_name") or "-"
    role = get_role(uid)
    summary = (
        "👤 Participant card\n"
        f"ID: {uid}\n"
        f"Name: {full_name}\n"
        f"Username: {username}\n"
        f"Role: {role}\n"
        f"First interaction: {pdata.get('first_seen', '-')}\n"
        f"Last interaction: {pdata.get('last_seen', '-')}\n"
        f"Last action: {pdata.get('last_action', '-')}\n"
        f"Total actions: {pdata.get('interaction_count', 0)}\n\n"
        "Recent actions:\n"
        f"{recent_user_activity(uid)}"
    )
    await call.message.answer(summary, reply_markup=participant_actions_kb(uid))
    await call.answer()


@dp.callback_query(F.data.startswith("admin_request_view:"))
async def cb_admin_request_view(call: CallbackQuery):
    if not await ensure_admin_callback(call): return
    uid_text = call.data.split(":", 1)[1]
    if not uid_text.isdigit():
        await call.answer("Invalid user_id", show_alert=True)
        return
    req = ACCESS.get("access_requests", {}).get(uid_text)
    if not req or req.get("status") != "pending":
        await call.answer("The request has already been processed or does not exist.", show_alert=True)
        return
    username = f"@{req.get('username')}" if req.get("username") else "-"
    summary = (
        "📝 Request card\n"
        f"ID: {uid_text}\n"
        f"Name: {req.get('full_name', '-')}\n"
        f"Username: {username}\n"
        f"Status: {req.get('status', '-') }\n"
        f"Submitted: {req.get('submitted_at', '-') }\n"
        f"Comment: {req.get('note', '-') or '-'}\n\n"
        "Choose an action below."
    )
    await call.message.answer(summary, reply_markup=request_actions_kb(int(uid_text)))
    await call.answer()


@dp.callback_query(F.data.startswith("admin_request_add:"))
async def cb_admin_request_add(call: CallbackQuery):
    if not await ensure_admin_callback(call): return
    uid_text = call.data.split(":", 1)[1]
    if not uid_text.isdigit():
        await call.answer("Invalid user_id", show_alert=True)
        return
    uid = int(uid_text)
    await call.message.answer(
        f"Choose a role for user {uid}:",
        reply_markup=request_role_selection_kb(uid),
    )
    await call.answer()


@dp.callback_query(F.data.startswith("admin_request_add_role:"))
async def cb_admin_request_add_role(call: CallbackQuery):
    if not await ensure_admin_callback(call):
        return
    parts = call.data.split(":")
    if len(parts) != 3:
        await call.answer("Invalid request data", show_alert=True)
        return
    _, uid_text, role_text = parts
    if not uid_text.isdigit():
        await call.answer("Invalid user_id", show_alert=True)
        return
    role = normalize_role(role_text)
    if role not in ROLE_SET:
        await call.answer("Invalid role", show_alert=True)
        return
    uid = int(uid_text)
    sync_user_access(uid, role)
    resolve_access_request(uid, "approved")
    save_access(ACCESS)
    await call.message.answer(f"✅ User {uid} added to active participants with role {role}.")
    await call.answer("User added")


@dp.callback_query(F.data.startswith("admin_request_delete:"))
async def cb_admin_request_delete(call: CallbackQuery):
    if not await ensure_admin_callback(call): return
    uid_text = call.data.split(":", 1)[1]
    if not uid_text.isdigit():
        await call.answer("Invalid user_id", show_alert=True)
        return
    uid = int(uid_text)
    await call.message.answer(
        f"Confirm deletion of the request from user {uid}?",
        reply_markup=confirm_request_delete_kb(uid),
    )
    await call.answer()


@dp.callback_query(F.data.startswith("admin_request_delete_confirm:"))
async def cb_admin_request_delete_confirm(call: CallbackQuery):
    if not await ensure_admin_callback(call):
        return
    uid_text = call.data.split(":", 1)[1]
    if not uid_text.isdigit():
        await call.answer("Invalid user_id", show_alert=True)
        return
    uid = int(uid_text)
    resolve_access_request(uid, "denied")
    save_access(ACCESS)
    await call.message.answer(f"🗑 Request from user {uid} removed from the pending list.")
    await call.answer("Request deleted")


@dp.callback_query(F.data.startswith("admin_remove_participant:"))
async def cb_admin_remove_participant(call: CallbackQuery):
    if not await ensure_admin_callback(call): return
    uid_text = call.data.split(":", 1)[1]
    if not uid_text.isdigit():
        await call.answer("Invalid user_id", show_alert=True)
        return
    uid = int(uid_text)
    if is_primary_admin_id(uid):
        await call.answer("The primary administrator cannot be removed.", show_alert=True)
        return
    await call.message.answer(
        f"Confirm removal of participant {uid}?",
        reply_markup=confirm_participant_delete_kb(uid),
    )
    await call.answer()


@dp.callback_query(F.data.startswith("admin_remove_participant_confirm:"))
async def cb_admin_remove_participant_confirm(call: CallbackQuery):
    if not await ensure_admin_callback(call):
        return
    uid_text = call.data.split(":", 1)[1]
    if not uid_text.isdigit():
        await call.answer("Invalid user_id", show_alert=True)
        return
    uid = int(uid_text)
    if is_primary_admin_id(uid):
        await call.answer("The primary administrator cannot be removed.", show_alert=True)
        return
    sync_user_access(uid, None)
    save_access(ACCESS)
    await call.message.answer(f"🗑 User {uid} removed from active participants.")
    await call.answer("Participant removed")

# Grant access → ask for user_id
@dp.callback_query(F.data == "admin_grant")
async def cb_admin_grant(call: CallbackQuery, state: FSMContext):
    if not await ensure_admin_callback(call): return
    await state.set_state(AdminForm.grant_user)
    await call.message.answer("Send the user_id of the user to grant access to, then choose a role with the buttons.")
    await call.answer()

@dp.message(AdminForm.grant_user)
async def on_admin_grant_user(message: Message, state: FSMContext):
    if not await ensure_admin_message(message): return
    txt = (message.text or "").strip()
    if not txt.isdigit():
        await message.answer("Enter a numeric user_id.")
        return
    uid = int(txt)
    await state.update_data(target_uid=txt)
    await state.set_state(AdminForm.role_value)
    await message.answer(
        f"ID {uid} accepted. Now choose a role:",
        reply_markup=role_selection_kb(),
    )

# Revoke access → ask for user_id
@dp.callback_query(F.data == "admin_deny")
async def cb_admin_deny(call: CallbackQuery, state: FSMContext):
    if not await ensure_admin_callback(call): return
    await state.set_state(AdminForm.deny_user)
    await call.message.answer("Send the user_id of the user whose access should be revoked.")
    await call.answer()

@dp.message(AdminForm.deny_user)
async def on_admin_deny_user(message: Message, state: FSMContext):
    if not await ensure_admin_message(message): return
    txt = (message.text or "").strip()
    if not txt.isdigit():
        await message.answer("Enter a numeric user_id.")
        return
    uid = int(txt)
    if is_primary_admin_id(uid):
        await message.answer("⛔ The primary administrator cannot be removed.")
        await state.clear()
        return
    sync_user_access(uid, None)
    resolve_access_request(uid, "denied")
    save_access(ACCESS)
    await message.answer(f"🚫 Access for user {uid} fully revoked.")
    await state.clear()

# Assign role → first user_id, then role
@dp.callback_query(F.data == "admin_role")
async def cb_admin_role(call: CallbackQuery, state: FSMContext):
    if not await ensure_admin_callback(call): return
    await state.set_state(AdminForm.role_user)
    await call.message.answer("Send the user_id of the user to assign a role to (viewer|operator|model|admin).")
    await call.answer()

@dp.message(AdminForm.role_user)
async def on_admin_role_user(message: Message, state: FSMContext):
    if not await ensure_admin_message(message): return
    txt = (message.text or "").strip()
    if not txt.isdigit():
        await message.answer("Enter a numeric user_id.")
        return
    await state.update_data(target_uid=txt)
    await state.set_state(AdminForm.role_value)
    await message.answer("Choose a role:", reply_markup=role_selection_kb())


@dp.callback_query(F.data.startswith("admin_set_role:"))
async def cb_admin_set_role(call: CallbackQuery, state: FSMContext):
    if not await ensure_admin_callback(call):
        return
    role = normalize_role(call.data.split(":", 1)[1])
    if role not in ROLE_SET:
        await call.answer("Invalid role", show_alert=True)
        return
    data = await state.get_data()
    target_uid = data.get("target_uid")
    if not target_uid:
        await call.answer("Specify the user_id first.", show_alert=True)
        return
    uid = int(target_uid)
    if is_primary_admin_id(uid) and role != "admin":
        await call.message.answer("⛔ The primary administrator always remains admin.")
        await state.clear()
        await call.answer()
        return
    sync_user_access(uid, role)
    resolve_access_request(uid, "approved")
    save_access(ACCESS)
    await call.message.answer(f"✅ Role for user {uid} set to: {role}")
    await state.clear()
    await call.answer("Role saved")

@dp.message(AdminForm.role_value)
async def on_admin_role_value(message: Message, state: FSMContext):
    if not await ensure_admin_message(message): return
    role = normalize_role((message.text or "").strip())
    if role not in ROLE_SET:
        await message.answer("Role must be one of: viewer | operator | model | admin")
        return
    data = await state.get_data()
    target_uid = data.get("target_uid")
    if not target_uid:
        await message.answer("Specify the user_id first, then the role.")
        await state.clear()
        return
    uid = int(target_uid)
    if is_primary_admin_id(uid) and role != "admin":
        await message.answer("⛔ The primary administrator always remains admin.")
        await state.clear()
        return
    sync_user_access(uid, role)
    resolve_access_request(uid, "approved")
    save_access(ACCESS)
    await message.answer(f"✅ Role for user {uid} set to: {role}")
    await state.clear()

# -----------------------------
# MENU PRESETS (simple lists, no code)
# -----------------------------
AVAILABLE_SYMBOLS: List[str] = cfg.get("available_symbols", [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "LTC/USDT", "SOL/USDC", 
])
AVAILABLE_EXCHANGES: List[str] = cfg.get("available_exchanges", [
    "binance", "kucoin", "bybit", "okx"
])
AVAILABLE_TIMEFRAMES: List[str] = cfg.get("available_timeframes", [
    "1m", "5m", "15m", "30m", "1h", "4h", "1d"
])
AVAILABLE_HISTORY_DAYS: List[int] = cfg.get("available_history_days", [1, 3, 7, 30, 90, 180, 365, 730, 1095])
MAX_SYMBOLS_PER_USER: int = int(cfg.get("max_symbols_per_user", 6))

# -----------------------------
# PANELS
# -----------------------------

def panel_button(text: str, callback_data: str, enabled: bool) -> InlineKeyboardButton:
    if enabled:
        return InlineKeyboardButton(text=text, callback_data=callback_data)
    return InlineKeyboardButton(text=f"🔒 {text}", callback_data=f"locked:{callback_data}")


def build_panel(user_id: Optional[int]) -> InlineKeyboardMarkup:
    is_admin = is_admin_id(user_id)
    can_session = has_session_access(user_id)
    can_model = has_model_access(user_id)
    rows = [
        [panel_button("⏹ Stop", "stop", is_admin)],
        [
            panel_button("🟢 Status", "status", is_admin),
            panel_button("🧾 Log", "lastlog", is_admin),
        ],
        [
            panel_button("📈 Global plots", "plots", is_admin),
        ],
        [
            panel_button("📋 Settings menu", "menu", can_session),
        ],
        [
            panel_button("👤 My session: start", "my_run", can_session),
            panel_button("🛑 My session: stop", "my_stop", can_session),
        ],
        [
            panel_button("🧾 My log", "my_log", can_session),
            panel_button("📄 My report", "my_report", can_session),
        ],
        [
            panel_button("🔄 Reset process", "my_reset", can_session),
            panel_button("ℹ️ My config", "my_config", can_session),
            panel_button("📦 Download session", "my_download", can_session),
        ],
        [
            panel_button("🧠 merged.csv", "my_dataset", can_model),
        ],
    ]
    if is_admin:
        rows.append([InlineKeyboardButton(text="👑 Admin panel", callback_data="admin_open")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔢 Select pairs", callback_data="m_pairs")],
        [InlineKeyboardButton(text="🏦 Select exchanges", callback_data="m_exchanges")],
        [InlineKeyboardButton(text="🕒 Select timeframe", callback_data="m_timeframe")],
        [InlineKeyboardButton(text="📆 History period (days)", callback_data="m_days")],
        [InlineKeyboardButton(text="⬅️ Back", callback_data="back_panel")],
    ])


def build_list_keyboard(items: List[str], prefix: str, selected: List[str] | None = None, cols: int = 2) -> InlineKeyboardMarkup:
    selected = selected or []
    rows = []
    row = []
    for i, it in enumerate(items):
        mark = "✅ " if it in selected else ""  # mark the selected item
        row.append(InlineKeyboardButton(text=f"{mark}{it}", callback_data=f"{prefix}:{i}"))
        if len(row) >= cols:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton(text="⬅️ Back", callback_data="menu")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_days_keyboard(days_options: List[int], selected: int | None, collection_mode: str = "auto") -> InlineKeyboardMarkup:
    rows = []
    row = []
    for i, d in enumerate(days_options):
        mark = "✅ " if (selected == d and collection_mode != "full") else ""
        row.append(InlineKeyboardButton(text=f"{mark}{d}", callback_data=f"days:{i}"))
        if len(row) >= 4:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    # "Entire history" button
    full_mark = "✅ " if collection_mode == "full" else ""
    rows.append([InlineKeyboardButton(text=f"{full_mark}📚 Entire history (full)", callback_data="days:full")])
    # Manual input button
    rows.append([InlineKeyboardButton(text="⌨️ Enter number of days manually", callback_data="days:custom_input")])
    rows.append([InlineKeyboardButton(text="⬅️ Back", callback_data="menu")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def format_uptime(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# -----------------------------
# GLOBAL PROCESS (unchanged)
# -----------------------------
async def start_pipeline(message: Message):
    global proc, started_at
    # Reset if the process is not actually running
    if proc and not _is_process_alive(proc):
        proc = None
    if proc and _is_process_alive(proc):
        await message.answer(f"⚠️ Already running: PID {proc.pid}")
        return

    stdout_path = ROOT / "pipeline_stdout.log"
    stderr_path = ROOT / "pipeline_stderr.log"
    stdout_f = open(stdout_path, "ab", buffering=0)
    stderr_f = open(stderr_path, "ab", buffering=0)

    p = psutil.Popen(
        PIPE_CMD,
        stdout=stdout_f,
        stderr=stderr_f,
        cwd=ROOT
    )
    proc = p
    started_at = time.time()
    await message.answer(f"▶️ Pipeline started. PID={p.pid}")

async def stop_pipeline(message: Message):
    global proc
    if not proc or not _is_process_alive(proc):
        proc = None
        await message.answer("ℹ️ Pipeline is not running.")
        return
    try:
        os.kill(proc.pid, signal.SIGINT)
        gone, alive = psutil.wait_procs([proc], timeout=8)
        if alive:
            os.kill(proc.pid, signal.SIGTERM)
        await message.answer("⏹️ Stopped.")
    except Exception as e:
        await message.answer(f"❌ Failed to stop: {e}")
    finally:
        proc = None

async def status_pipeline(message: Message):
    # Reset proc if it is stale
    global proc
    if proc and not _is_process_alive(proc):
        proc = None
    if proc and _is_process_alive(proc):
        cpu = proc.cpu_percent(interval=0.2)
        mem = proc.memory_info().rss / (1024 * 1024)
        up = format_uptime(time.time() - started_at)
        await message.answer(
            "🟢 Running\n"
            f"PID: {proc.pid}\n"
            f"Uptime: {up}\n"
            f"CPU: {cpu:.1f}%\n"
            f"RAM: {mem:.1f} MB"
        )
    else:
        await message.answer("🔴 Not running")

async def tail_log(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return "(log not created yet)"
    with open(path, "rb") as f:
        try:
            f.seek(-8192, os.SEEK_END)
        except OSError:
            f.seek(0)
        chunk = f.read().decode("utf-8", errors="ignore")
    return "\n".join(chunk.strip().splitlines()[-lines:])


def get_session_log_path(s: "Session") -> Path:
    return Path(s.config.get("log_file") or (s.root / "crypto_log.txt"))


def get_session_merged_csv_path(s: "Session") -> Path:
    base_path = Path(s.config.get("base_path") or s.root)
    merged_file = s.config.get("merged_file")
    if merged_file:
        return Path(merged_file)
    return base_path / "full_data_base" / "merged.csv"


def get_latest_user_session(uid: int) -> "Session":
    latest_dir = find_latest_session_dir(uid)
    if latest_dir:
        return Session(uid, root=latest_dir)
    return SESS.get(uid)


def extract_latest_report(log_text: str) -> str:
    lines = log_text.splitlines()
    start_idx = -1
    for idx, line in enumerate(lines):
        # Take only the actual report header, not the line about sending it to Telegram.
        if "✅ Final report:" in line and "📬 Report sent" not in line:
            start_idx = idx
    if start_idx < 0:
        return ""

    report_lines = [lines[start_idx]]
    for line in lines[start_idx + 1:]:
        stripped = line.strip()
        # Summary items look like "- ..." without a timestamp; a new timestamped line ends the block.
        if stripped.startswith("["):
            break
        if not stripped:
            continue
        report_lines.append(line)
    return "\n".join(report_lines).strip()


async def send_own_session_log(message: Message, uid: int | None = None, lines: int = 80) -> None:
    effective_uid = uid if uid is not None else message.from_user.id
    s = get_latest_user_session(effective_uid)
    log_path = get_session_log_path(s)
    txt = await tail_log(log_path, lines)
    if not txt:
        txt = "(log is empty)"
    if len(txt) > 4000:
        txt = txt[-4000:]
    await message.answer(f"🧾 Last lines of your log:\n{txt}")


async def send_own_final_report(message: Message, uid: int | None = None) -> None:
    effective_uid = uid if uid is not None else message.from_user.id
    s = get_latest_user_session(effective_uid)
    log_path = get_session_log_path(s)
    if not log_path.exists():
        await message.answer("ℹ️ The final report is not available yet: the current session log has not been created.")
        return
    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    report = extract_latest_report(log_text)
    if not report:
        await message.answer("ℹ️ The final report was not found yet. The session is probably still running or has not reached the end.")
        return
    if len(report) > 4000:
        report = report[-4000:]
    await message.answer("📄 Your final report:\n" + report)


async def send_model_dataset(message: Message, uid: int | None = None) -> None:
    effective_uid = uid if uid is not None else message.from_user.id
    s = get_latest_user_session(effective_uid)
    dataset_path = get_session_merged_csv_path(s)
    if not dataset_path.exists():
        await message.answer("ℹ️ merged.csv is not available yet. Wait for the session to finish and the final dataset to be built.")
        return
    await message.answer_document(FSInputFile(dataset_path), caption="🧠 Final dataset merged.csv")


async def notify_user_pipeline_finished(chat_id: int, user_id: int, session_root: Path, pid: int) -> None:
    """Notify quickly when report appears, then send process-finished status."""
    log_path = session_root / "crypto_log.txt"
    report_sent = False
    exit_code: int | None = None

    try:
        proc_ref = psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        proc_ref = None
        exit_code = 0

    # Send the report as soon as it appears in the log, without waiting for all of the pipeline's Telegram sends to finish.
    while True:
        if not report_sent and log_path.exists():
            try:
                report = extract_latest_report(log_path.read_text(encoding="utf-8", errors="ignore"))
                if report:
                    await bot.send_message(chat_id, "📄 Final session report:\n" + report[:3900])
                    report_sent = True
            except Exception:
                pass

        if proc_ref is None:
            break

        try:
            exit_code = await asyncio.to_thread(proc_ref.wait, timeout=1.0)
            break
        except psutil.TimeoutExpired:
            await asyncio.sleep(0.6)
            continue
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            exit_code = 0
            break
        except Exception:
            exit_code = None
            break

    # If the process exited before the loop caught the report, try once more.
    if not report_sent and log_path.exists():
        try:
            report = extract_latest_report(log_path.read_text(encoding="utf-8", errors="ignore"))
            if report:
                await bot.send_message(chat_id, "📄 Final session report:\n" + report[:3900])
        except Exception:
            pass

    status = "✅ Data collection finished." if exit_code in (0, None) else f"⚠️ Data collection finished with exit code: {exit_code}."
    footer = (
        "\nAvailable: /my_report (summary), /my_log (log), /my_download (session archive)."
        + (" /my_dataset (merged.csv)." if has_model_access(user_id) else "")
    )
    try:
        await bot.send_message(chat_id, f"{status}\nPID={pid}{footer}")
    except Exception:
        return

async def send_lastlog(message: Message):
    txt = await tail_log(LOG_FILE, 80)
    if not txt:
        txt = "(log is empty)"
    if len(txt) > 4000:
        txt = txt[-4000:]
    await message.answer(f"🧾 Last log lines:\n{txt}")

async def send_plots(message: Message, limit: int = 6):
    if not PLOT_PATH.exists():
        await message.answer("Plot folder not found.")
        return
    pngs = sorted((p for p in PLOT_PATH.glob("**/*.png")),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not pngs:
        await message.answer("No plots found.")
        return
    for p in pngs[:limit]:
        try:
            await message.answer_document(FSInputFile(p), caption=f"📈 {p.name}")
        except Exception as e:
            await message.answer(f"⚠️ Failed to send {p.name}: {e}")

# -----------------------------
# PERSONAL SESSIONS (the base YAML is left untouched)
# -----------------------------
BASE_CONFIG = cfg
SESSIONS_ROOT = (ROOT.parent / "sessions").resolve()
SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)

class Session:
    def __init__(self, user_id: int, root: Path | None = None):
        self.user_id = user_id
        # If root is given, use it (to restore the latest session)
        if root:
            self.root = root
        else:
            self.root = SESSIONS_ROOT / str(user_id) / datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.root.mkdir(parents=True, exist_ok=True)
        self.config_path = self.root / "config.yaml"
        self.proc: psutil.Popen | None = None
        self.started_at = 0.0
        self.keep = False

        if self.config_path.exists():
            # Load the existing configuration
            try:
                self.config = yaml.safe_load(self.config_path.read_text()) or {}
            except Exception:
                self.config = {}
        else:
            # New config: base paths + empty required fields
            self.config = dict(BASE_CONFIG)
            self.config["symbols"] = []
            self.config["exchanges"] = []
            self.config["timeframe"] = None
            self.config["history_days"] = None
            self.config["collection_mode"] = "auto"

        # Always pin paths inside the user/session root so that different users' data never overlaps.
        self._apply_isolated_paths()
        self.save()

        (self.root / "data").mkdir(exist_ok=True)
        (self.root / "plots").mkdir(exist_ok=True)

    def save(self):
        with open(self.config_path, "w") as f:
            yaml.safe_dump(self.config, f, allow_unicode=True)

    def reload(self):
        if self.config_path.exists():
            try:
                loaded = yaml.safe_load(self.config_path.read_text()) or {}
                # Reassign only the contents so that references in the code stay valid
                self.config = loaded
                self._apply_isolated_paths()
            except Exception:
                pass

    def _apply_isolated_paths(self):
        self.config["base_path"] = str(self.root)
        self.config["data_path"] = str(self.root / "data")
        self.config["plot_path"] = str(self.root / "plots")
        self.config["log_file"] = str(self.root / "crypto_log.txt")
        self.config["archive_path"] = str(self.root / "archive")
        self.config["merged_file"] = str(self.root / "full_data_base" / "merged.csv")

class SessionManager:
    def __init__(self):
        self.sessions: dict[int, Session] = {}

    def get(self, uid: int) -> Session:
        if uid not in self.sessions:
            # Try to restore the user's latest session, if one exists
            user_dir = SESSIONS_ROOT / str(uid)
            latest = None
            if user_dir.exists():
                candidates = sorted([p for p in user_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
                if candidates:
                    latest = candidates[-1]
            if latest and (latest / "config.yaml").exists():
                self.sessions[uid] = Session(uid, root=latest)
            else:
                self.sessions[uid] = Session(uid)
        return self.sessions[uid]

    def clear(self, uid: int):
        if uid in self.sessions:
            del self.sessions[uid]

SESS = SessionManager()

# Helper for checking a process
def _is_process_alive(proc: psutil.Popen | None) -> bool:
    """Reliable check that the process is alive (exists and is not a zombie)."""
    if proc is None:
        return False
    try:
        # Check that the process exists
        if not psutil.pid_exists(proc.pid):
            return False
        # poll() updates returncode
        proc.poll()
        if proc.returncode is not None:
            return False
        # Final is_running check
        return proc.is_running()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False
    except Exception:
        return False

# start/stop of the personal pipeline
async def start_user_pipeline(message: Message, user_id: int | None = None):
    uid = user_id or message.from_user.id
    s = SESS.get(uid)
    s.reload()
    s.save()
    missing = validate_session_config(s)
    if missing:
        await message.answer(
            "⚠️ Missing parameters to start:\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\n\nCurrent configuration:\n"
            + human_config_summary(s)
            + f"debug: uid={uid}, config_path={s.config_path}"
        )
        return
    # Improved check: reset proc if the process is not actually running
    if s.proc and not _is_process_alive(s.proc):
        s.proc = None
    if s.proc and _is_process_alive(s.proc):
        await message.answer(f"⚠️ Your session is already running: PID {s.proc.pid}")
        return

    stdout_path = s.root / "stdout.log"
    stderr_path = s.root / "stderr.log"
    stdout_f = open(stdout_path, "ab", buffering=0)
    stderr_f = open(stderr_path, "ab", buffering=0)

    env = os.environ.copy()
    env["PIPELINE_CONFIG"] = str(s.config_path)
    env["SESSION_ROOT"] = str(s.root)
    env["TELEGRAM_EXTRA_CHAT_ID"] = str(uid)
    env["PIPELINE_SEND_DATASET"] = "1" if has_model_access(uid) else "0"

    p = psutil.Popen(
        PIPE_CMD,
        stdout=stdout_f,
        stderr=stderr_f,
        cwd=ROOT,
        env=env,
    )
    s.proc = p
    s.started_at = time.time()
    await message.answer("▶️ Session started\n" + human_config_summary(s, with_root=False) + f"PID={p.pid}")
    await message.answer("⏳ Collecting data. This may take a while.")
    chat_id = message.chat.id if message.chat else uid
    asyncio.create_task(notify_user_pipeline_finished(chat_id, uid, s.root, p.pid))

async def stop_user_pipeline(message: Message, keep: bool = False, user_id: int | None = None):
    uid = user_id or message.from_user.id
    s = SESS.get(uid)
    s.reload()
    # Reset proc if the process is stale
    if s.proc and not _is_process_alive(s.proc):
        s.proc = None
    if not s.proc or not _is_process_alive(s.proc):
        await message.answer("ℹ️ Your session is not running.")
        return
    try:
        os.kill(s.proc.pid, signal.SIGINT)
        gone, alive = psutil.wait_procs([s.proc], timeout=8)
        if alive:
            os.kill(s.proc.pid, signal.SIGTERM)
        await message.answer("⏹️ Your session has been stopped.")
    except Exception as e:
        await message.answer(f"❌ Failed to stop: {e}")
    finally:
        s.proc = None
        if not keep:
            s.keep = False

async def status_user_pipeline(message: Message):
    uid = message.from_user.id
    s_obj = SESS.get(uid)
    # Reset proc if the process is stale
    if s_obj.proc and not _is_process_alive(s_obj.proc):
        s_obj.proc = None
    if s_obj.proc and _is_process_alive(s_obj.proc):
        cpu = s_obj.proc.cpu_percent(interval=0.2)
        mem = s_obj.proc.memory_info().rss / (1024 * 1024)
        up = format_uptime(time.time() - s_obj.started_at)
        await message.answer(
            "🟢 Your session is running\n"
            + human_config_summary(s_obj)
            + f"Uptime: {up}\nCPU: {cpu:.1f}%\nRAM: {mem:.1f} MB"
        )
    else:
        await message.answer("🔴 Your session is not running.")

# HUMAN SUMMARY (no code/YAML)

def human_config_summary(s: Session, with_root: bool = False) -> str:
    syms = ", ".join(s.config.get("symbols") or [])
    exs  = ", ".join(s.config.get("exchanges") or [])
    tf   = s.config.get("timeframe") or "-"
    mode = s.config.get("collection_mode", "auto")
    days_val = s.config.get("history_days")
    if mode == "full":
        period_str = "📚 Entire history (full)"
    elif days_val is not None:
        period_str = f"{days_val} days back from today"
    else:
        period_str = "-"
    lines = [
        f"Pairs: {syms or '-'}",
        f"Exchanges: {exs or '-'}",
        f"Timeframe: {tf}",
        f"Period: {period_str}",
    ]
    if with_root:
        lines.append(f"root: {s.root}")
    return "\n".join(lines) + "\n"


def validate_session_config(s: Session) -> list[str]:
    errors = []
    if not s.config.get("symbols"):
        errors.append("Select pairs in the menu.")
    if not s.config.get("exchanges"):
        errors.append("Select exchanges in the menu.")
    if not s.config.get("timeframe"):
        errors.append("Select a timeframe in the menu.")
    mode = s.config.get("collection_mode", "auto")
    if mode != "full" and s.config.get("history_days") is None:
        errors.append("Select the history period (days) in the menu.")
    return errors

# user's latest session (if config.yaml exists)
def find_latest_session_dir(uid: int) -> Path | None:
    user_dir = SESSIONS_ROOT / str(uid)
    if not user_dir.exists():
        return None
    dirs = sorted([p for p in user_dir.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
    for d in dirs:
        if (d / "config.yaml").exists():
            return d
    return None


# -----------------------------
# COMMANDS/BUTTONS
# -----------------------------

def _args(text: str) -> str:
    parts = text.split(maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else ""

@dp.message(Command("start"))
async def cmd_start(message: Message):
    if not await ensure_panel_message(message): return
    await message.answer("Hi! This is the pipeline control panel. Choose an action using the buttons or commands:",reply_markup=build_panel(message.from_user.id)
                         )

@dp.message(Command("panel"))
async def cmd_panel(message: Message):
    if not await ensure_panel_message(message): return
    await message.answer("Control panel:", reply_markup=build_panel(message.from_user.id))

@dp.message(Command("menu"))
async def cmd_menu(message: Message):
    if not await ensure_operator_message(message): return
    await message.answer("Settings menu (no code required)", reply_markup=build_menu())

# Global
@dp.message(Command("run"))
async def cmd_run(message: Message):
    if not await ensure_admin_message(message): return
    await message.answer("Global run is disabled. Use 'My session: start' or the /my_run command.")

@dp.message(Command("stop"))
async def cmd_stop(message: Message):
    if not await ensure_admin_message(message): return
    await stop_pipeline(message)

@dp.message(Command("status"))
async def cmd_status(message: Message):
    if not await ensure_admin_message(message): return
    await status_pipeline(message)

@dp.message(Command("lastlog"))
async def cmd_lastlog(message: Message):
    if not await ensure_admin_message(message): return
    await send_lastlog(message)

@dp.message(Command("plots"))
async def cmd_plots(message: Message):
    if not await ensure_admin_message(message): return
    await send_plots(message, limit=6)

@dp.message(Command("restart"))
async def cmd_restart(message: Message):
    if not await ensure_admin_message(message): return
    await message.answer("Global restart is disabled. Use 'My session: start' or /my_run.")

# Personal
@dp.message(Command("my_new"))
async def cmd_my_new(message: Message):
    if not await ensure_operator_message(message): return
    uid = message.from_user.id
    SESS.clear(uid)
    s = SESS.get(uid)
    await message.answer(
        "✅ Personal session is ready. Configure it via /menu or the \"📋 Settings menu\" button",
        reply_markup=build_menu()
    )

@dp.message(Command("my_config"))
async def cmd_my_config(message: Message):
    if not await ensure_operator_message(message): return
    s = SESS.get(message.from_user.id)
    await message.answer(human_config_summary(s))

@dp.message(Command("my_run"))
async def cmd_my_run(message: Message):
    if not await ensure_operator_message(message): return
    await start_user_pipeline(message)

@dp.message(Command("my_status"))
async def cmd_my_status(message: Message):
    if not await ensure_operator_message(message): return
    await status_user_pipeline(message)

@dp.message(Command("my_stop"))
async def cmd_my_stop(message: Message):
    if not await ensure_operator_message(message): return
    keep = _args(message.text).lower().strip() == "keep"
    await stop_user_pipeline(message, keep=keep)

@dp.message(Command("my_log"))
async def cmd_my_log(message: Message):
    if not await ensure_operator_message(message): return
    await send_own_session_log(message)

@dp.message(Command("my_report"))
async def cmd_my_report(message: Message):
    if not await ensure_operator_message(message): return
    await send_own_final_report(message)

@dp.message(Command("my_dataset"))
async def cmd_my_dataset(message: Message):
    if not await ensure_model_message(message): return
    await send_model_dataset(message)

@dp.message(Command("my_reset"))
async def cmd_my_reset(message: Message):
    """Force-reset the session process state (if it is stuck)."""
    if not await ensure_operator_message(message): return
    uid = message.from_user.id
    s = SESS.get(uid)
    old_proc = s.proc
    s.proc = None
    if old_proc:
        # Try to kill the process if it exists
        try:
            if psutil.pid_exists(old_proc.pid):
                os.kill(old_proc.pid, signal.SIGKILL)
        except Exception:
            pass
        await message.answer(f"🔄 Session process reset (was PID={old_proc.pid}). You can start it again now.")
    else:
        await message.answer("🔄 Process state reset. You can start the session now.")

@dp.message(Command("download"))
async def cmd_download(message: Message):
    if not await ensure_operator_message(message): return
    s = SESS.get(message.from_user.id)
    summary = human_config_summary(s)
    (s.root / "SUMMARY.txt").write_text(summary, encoding="utf-8")
    zpath = s.root / "session_bundle.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in [s.root / "data", s.root / "plots"]:
            if p.exists():
                for f in p.rglob("*"):
                    if f.is_file():
                        z.write(f, arcname=f.relative_to(s.root))
        z.write(s.root / "SUMMARY.txt", arcname="SUMMARY.txt")
    await message.answer_document(FSInputFile(zpath), caption="📦 Current session archive (data + plots + SUMMARY)")


# ====== CALLBACKS ======
# Panel
@dp.callback_query(F.data == "menu")
async def cb_menu(call: CallbackQuery):
    if not await ensure_operator_callback(call): return
    await call.message.edit_text("Settings menu (no code required)", reply_markup=build_menu())

@dp.callback_query(F.data == "back_panel")
async def cb_back_panel(call: CallbackQuery):
    if not await ensure_panel_callback(call): return
    await call.message.edit_text("Control panel:", reply_markup=build_panel(call.from_user.id))

@dp.callback_query(F.data.startswith("locked:"))
async def cb_locked(call: CallbackQuery):
    if not await ensure_panel_callback(call):
        return
    role = get_role(call.from_user.id)
    if role == "viewer":
        await call.answer("You have the viewer role: the panel is read-only.", show_alert=True)
        return
    await call.answer("This feature is not available for your role yet.", show_alert=True)

@dp.callback_query(F.data == "guest_my_id")
async def cb_guest_my_id(call: CallbackQuery):
    if not await enforce_callback_rate_limit(call):
        return
    track_user_action(call.from_user, f"callback:{call.data}")
    uid = call.from_user.id if call.from_user else 0
    await call.message.answer(
        f"Your Telegram ID: {uid}\n"
        "If you need access, send this ID to an administrator or press the request-access button."
    )
    await call.answer()

@dp.callback_query(F.data == "guest_apply_access")
async def cb_guest_apply_access(call: CallbackQuery):
    if not await enforce_callback_rate_limit(call):
        return
    track_user_action(call.from_user, f"callback:{call.data}")
    ok, text = request_access(call.from_user)
    if ok:
        await notify_admin_about_request(call.from_user)
    await call.message.answer(text)
    await call.answer("Request sent" if ok else "Request already exists", show_alert=not ok)

# Menu: pairs
@dp.callback_query(F.data == "m_pairs")
async def cb_m_pairs(call: CallbackQuery):
    if not await ensure_operator_callback(call): return
    s = SESS.get(call.from_user.id)
    kb = build_list_keyboard(AVAILABLE_SYMBOLS, prefix="pair", selected=s.config.get("symbols") or [], cols=2)
    await call.message.edit_text("Select pairs (tap to toggle on/off).", reply_markup=kb)

@dp.callback_query(F.data.startswith("pair:"))
async def cb_pair_toggle(call: CallbackQuery):
    if not await ensure_operator_callback(call): return
    s = SESS.get(call.from_user.id)
    idx = int(call.data.split(":")[1])
    sym = AVAILABLE_SYMBOLS[idx]
    cur = s.config.get("symbols") or []
    if sym in cur:
        cur = [x for x in cur if x != sym]
    else:
        if len(cur) >= MAX_SYMBOLS_PER_USER:
            await call.answer(f"Max pairs: {MAX_SYMBOLS_PER_USER}", show_alert=True)
            return
        cur.append(sym)
    s.config["symbols"] = cur
    s.save()
    kb = build_list_keyboard(AVAILABLE_SYMBOLS, prefix="pair", selected=cur, cols=2)
    await call.message.edit_reply_markup(reply_markup=kb)
    await call.answer()

# Menu: exchanges
@dp.callback_query(F.data == "m_exchanges")
async def cb_m_exchanges(call: CallbackQuery):
    if not await ensure_operator_callback(call): return
    s = SESS.get(call.from_user.id)
    kb = build_list_keyboard(AVAILABLE_EXCHANGES, prefix="ex", selected=s.config.get("exchanges") or [], cols=2)
    await call.message.edit_text("Select exchanges (tap to toggle on/off).", reply_markup=kb)

@dp.callback_query(F.data.startswith("ex:"))
async def cb_ex_toggle(call: CallbackQuery):
    if not await ensure_operator_callback(call): return
    s = SESS.get(call.from_user.id)
    idx = int(call.data.split(":")[1])
    ex = AVAILABLE_EXCHANGES[idx]
    cur = s.config.get("exchanges") or []
    if ex in cur:
        cur = [x for x in cur if x != ex]
    else:
        cur.append(ex)
    s.config["exchanges"] = cur
    s.save()
    kb = build_list_keyboard(AVAILABLE_EXCHANGES, prefix="ex", selected=cur, cols=2)
    await call.message.edit_reply_markup(reply_markup=kb)
    await call.answer()

# Menu: timeframe
@dp.callback_query(F.data == "m_timeframe")
async def cb_m_timeframe(call: CallbackQuery):
    if not await ensure_operator_callback(call): return
    s = SESS.get(call.from_user.id)
    sel = s.config.get("timeframe")
    kb = build_list_keyboard(AVAILABLE_TIMEFRAMES, prefix="tf", selected=[sel] if sel else [], cols=4)
    await call.message.edit_text("Select a timeframe", reply_markup=kb)

@dp.callback_query(F.data.startswith("tf:"))
async def cb_tf_set(call: CallbackQuery):
    if not await ensure_operator_callback(call): return
    s = SESS.get(call.from_user.id)
    idx = int(call.data.split(":")[1])
    s.config["timeframe"] = AVAILABLE_TIMEFRAMES[idx]
    s.save()
    kb = build_list_keyboard(AVAILABLE_TIMEFRAMES, prefix="tf", selected=[s.config["timeframe"]], cols=4)
    await call.message.edit_reply_markup(reply_markup=kb)
    await call.answer()

# Menu: number of history days to fetch
@dp.callback_query(F.data == "m_days")
async def cb_m_days(call: CallbackQuery):
    if not await ensure_operator_callback(call): return
    s = SESS.get(call.from_user.id)
    sel = s.config.get("history_days")
    mode = s.config.get("collection_mode", "auto")
    kb = build_days_keyboard(AVAILABLE_HISTORY_DAYS, selected=sel, collection_mode=mode)
    await call.message.edit_text(
        "Select the history period (in days back from today)\n"
        "Or choose 'Entire history' to collect without limits.",
        reply_markup=kb
    )

@dp.callback_query(F.data == "days:full")
async def cb_days_full(call: CallbackQuery):
    if not await ensure_operator_callback(call): return
    s = SESS.get(call.from_user.id)
    s.config["collection_mode"] = "full"
    s.config["history_days"] = None  # not used in full mode
    s.save()
    kb = build_days_keyboard(AVAILABLE_HISTORY_DAYS, selected=None, collection_mode="full")
    await call.message.edit_reply_markup(reply_markup=kb)
    await call.answer("Entire history selected")

@dp.callback_query(F.data == "days:custom_input")
async def cb_days_custom_input(call: CallbackQuery, state: FSMContext):
    if not await ensure_operator_callback(call): return
    await state.set_state(DaysForm.custom_days)
    await call.message.answer("Enter the number of days (a number, e.g. 500):")
    await call.answer()

@dp.message(DaysForm.custom_days)
async def on_days_custom_input(message: Message, state: FSMContext):
    if not await ensure_operator_message(message): return
    txt = (message.text or "").strip()
    if not txt.isdigit() or int(txt) < 1:
        await message.answer("Enter a positive integer (e.g. 500).")
        return
    days_val = int(txt)
    s = SESS.get(message.from_user.id)
    s.config["history_days"] = days_val
    s.config["collection_mode"] = "auto"
    s.save()
    await message.answer(f"✅ History period set: {days_val} days (back from today)")
    await state.clear()

@dp.callback_query(F.data.startswith("days:"))
async def cb_days_set(call: CallbackQuery):
    if not await ensure_operator_callback(call): return
    # Skip special values - they are handled by dedicated handlers
    suffix = call.data.split(":")[1]
    if suffix in ("full", "custom_input"):
        return
    try:
        idx = int(suffix)
    except ValueError:
        await call.answer("Error: invalid index")
        return
    if idx < 0 or idx >= len(AVAILABLE_HISTORY_DAYS):
        await call.answer("Error: index out of range")
        return
    s = SESS.get(call.from_user.id)
    s.config["history_days"] = int(AVAILABLE_HISTORY_DAYS[idx])
    s.config["collection_mode"] = "auto"
    s.save()
    kb = build_days_keyboard(AVAILABLE_HISTORY_DAYS, selected=s.config["history_days"], collection_mode="auto")
    await call.message.edit_reply_markup(reply_markup=kb)
    await call.answer(f"Selected: {AVAILABLE_HISTORY_DAYS[idx]} days")

# admin management commands
@dp.message(Command("allow"))
async def cmd_allow(message: Message):
    if not await ensure_admin_message(message): return
    args = message.text.split()
    if len(args) < 2 or not args[1].isdigit():
        await message.answer("Usage: /allow <user_id> [viewer|operator|model|admin]")
        return
    uid = int(args[1])
    role = normalize_role(args[2]) if len(args) >= 3 else "viewer"
    if role not in ROLE_SET:
        await message.answer("Role must be one of: viewer | operator | model | admin")
        return
    sync_user_access(uid, role)
    resolve_access_request(uid, "approved")
    save_access(ACCESS)
    await message.answer(f"✅ User {uid} granted access: {role}")

@dp.message(Command("deny"))
async def cmd_deny(message: Message):
    if not await ensure_admin_message(message): return
    args = message.text.split()
    if len(args) < 2 or not args[1].isdigit():
        await message.answer("Usage: /deny <user_id>")
        return
    uid = int(args[1])
    if is_primary_admin_id(uid):
        await message.answer("⛔ The primary administrator cannot be removed.")
        return
    sync_user_access(uid, None)
    resolve_access_request(uid, "denied")
    save_access(ACCESS)
    await message.answer(f"🚫 Access for user {uid} revoked.")

@dp.message(Command("role"))
async def cmd_role(message: Message):
    if not await ensure_admin_message(message): return
    # /role <user_id> <viewer|operator|model|admin>
    args = message.text.split()
    if len(args) < 3 or not args[1].isdigit():
        await message.answer("Usage: /role <user_id> <viewer|operator|model|admin>")
        return
    uid = int(args[1])
    role = normalize_role(args[2])
    if role not in ROLE_SET:
        await message.answer("Role must be one of: viewer | operator | model | admin")
        return
    if is_primary_admin_id(uid) and role != "admin":
        await message.answer("⛔ The primary administrator always remains admin.")
        return
    sync_user_access(uid, role)
    resolve_access_request(uid, "approved")
    save_access(ACCESS)
    await message.answer(f"✅ Role for user {uid} set to: {role}")

@dp.message(Command("whoami", "id"))
async def cmd_whoami(message: Message):
    track_user_action(message.from_user, f"message:{message.text or '/whoami'}")
    uid = message.from_user.id if message.from_user else 0
    role = get_role(uid)
    await message.answer(
        f"Your Telegram ID: {uid}\n"
        f"Role: {role}\n"
        f"Admin: {'yes' if is_admin_id(uid) else 'no'}\n"
        f"Panel: {'yes' if has_panel_access(uid) else 'no'}\n"
        f"Session control: {'yes' if has_session_access(uid) else 'no'}\n"
        f"Model access: {'yes' if has_model_access(uid) else 'no'}\n\n"
        "If you need access, send this ID to an administrator."
    )

@dp.message(Command("ops"))
async def cmd_ops(message: Message):
    if not await ensure_admin_message(message): return
    roles = ACCESS.get("roles", {})
    body = "\n".join(
        f"{uid}: {normalize_role(role) or role}"
        for uid, role in sorted(roles.items(), key=lambda item: int(item[0]))
    ) if roles else "(empty)"
    await message.answer("Assigned roles:\n" + body)


# Buttons: global
@dp.callback_query(F.data == "run")
async def cb_run(call: CallbackQuery):
    if not await ensure_admin_callback(call): return
    await call.message.answer("Global run is disabled. Use 'My session: start' or /my_run.")
    await call.answer()

@dp.callback_query(F.data == "my_run")
async def cb_my_run(call: CallbackQuery):
    if not await ensure_operator_callback(call): return
    await start_user_pipeline(call.message, user_id=call.from_user.id)
    await call.answer()


@dp.callback_query(F.data == "stop")
async def cb_stop(call: CallbackQuery):
    if not await ensure_admin_callback(call): return
    await stop_pipeline(call.message)
    await call.answer()

@dp.callback_query(F.data == "status")
async def cb_status(call: CallbackQuery):
    if not await ensure_admin_callback(call): return
    await status_pipeline(call.message)
    await call.answer()

@dp.callback_query(F.data == "lastlog")
async def cb_lastlog(call: CallbackQuery):
    if not await ensure_admin_callback(call): return
    await send_lastlog(call.message)
    await call.answer()

@dp.callback_query(F.data == "plots")
async def cb_plots(call: CallbackQuery):
    if not await ensure_admin_callback(call): return
    await send_plots(call.message, limit=6)
    await call.answer()

@dp.callback_query(F.data == "restart")
async def cb_restart(call: CallbackQuery):
    if not await ensure_admin_callback(call): return
    await call.message.answer("Global restart is disabled. Use 'My session: start' or /my_run.")
    await call.answer("Unavailable")

@dp.callback_query(F.data == "my_stop")
async def cb_my_stop(call: CallbackQuery):
    if not await ensure_operator_callback(call): return
    await stop_user_pipeline(call.message, keep=False, user_id=call.from_user.id)
    await call.answer()

@dp.callback_query(F.data == "my_log")
async def cb_my_log(call: CallbackQuery):
    if not await ensure_operator_callback(call): return
    await send_own_session_log(call.message, uid=call.from_user.id)
    await call.answer()

@dp.callback_query(F.data == "my_report")
async def cb_my_report(call: CallbackQuery):
    if not await ensure_operator_callback(call): return
    await send_own_final_report(call.message, uid=call.from_user.id)
    await call.answer()

@dp.callback_query(F.data == "my_dataset")
async def cb_my_dataset(call: CallbackQuery):
    if not await ensure_model_callback(call): return
    await send_model_dataset(call.message, uid=call.from_user.id)
    await call.answer()

@dp.callback_query(F.data == "my_reset")
async def cb_my_reset(call: CallbackQuery):
    """Force-reset the session process state (if it is stuck)."""
    if not await ensure_operator_callback(call): return
    uid = call.from_user.id
    s = SESS.get(uid)
    old_proc = s.proc
    s.proc = None
    if old_proc:
        # Try to kill the process if it exists
        try:
            if psutil.pid_exists(old_proc.pid):
                os.kill(old_proc.pid, signal.SIGKILL)
        except Exception:
            pass
        await call.message.answer(f"🔄 Session process reset (was PID={old_proc.pid}). You can start it again now.")
    else:
        await call.message.answer("🔄 Process state reset. You can start the session now.")
    await call.answer()

@dp.callback_query(F.data == "my_config")
async def cb_my_config(call: CallbackQuery):
    if not await ensure_operator_callback(call): return
    s = SESS.get(call.from_user.id)
    await call.message.answer(human_config_summary(s))
    await call.answer()

@dp.callback_query(F.data == "my_download")
async def cb_my_download(call: CallbackQuery):
    if not await ensure_operator_callback(call): return
    s = SESS.get(call.from_user.id)
    (s.root / "SUMMARY.txt").write_text(human_config_summary(s), encoding="utf-8")
    zpath = s.root / "session_bundle.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in [s.root / "data", s.root / "plots"]:
            if p.exists():
                for f in p.rglob("*"):
                    if f.is_file():
                        z.write(f, arcname=f.relative_to(s.root))
        z.write(s.root / "SUMMARY.txt", arcname="SUMMARY.txt")
    await call.message.answer_document(FSInputFile(zpath), caption="📦 Current session archive (data + plots + SUMMARY)")
    await call.answer()

# -----------------------------
# MAIN
# -----------------------------
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
