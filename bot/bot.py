import logging
import asyncio
import traceback
import html
import json
import tempfile
import pydub
import base64
from io import BytesIO
from pathlib import Path
from datetime import datetime
import openai

import telegram
import shutil
from telegram import (
    Update,
    User,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    BotCommand
)
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackContext,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    AIORateLimiter,
    filters
)
from telegram.constants import ParseMode, ChatAction

import config
import database
import openai_utils
import io

# setup
db = database.Database()
logger = logging.getLogger(__name__)
FFMPEG_AVAILABLE = shutil.which('ffmpeg') is not None

user_semaphores = {}
user_tasks = {}

HELP_MESSAGE = """
👋 <b>Hello!</b> I'm your friendly <b>ChatGPT</b> bot 🤖

🔹 <b>/new</b> – Start a fresh conversation
🔄 <b>/retry</b> – Regenerate my last response
🎭 <b>/mode</b> – Switch chat mode (e.g. Assistant, Artist)
⚙️ <b>/settings</b> – View or update your preferences
💰 <b>/balance</b> – Check your usage balance
❓ <b>/help</b> – Show this help message

🖌️ In <b>Artist</b> mode you can generate images with DALL·E 3
🎨 In <b>GPT-Image Pro</b> you get advanced image editing tools
👥 Add me to group chats and mention me (@bot) to talk
🎤 Send voice messages and I'll transcribe them for you

Let’s get creative! 🌟
"""

HELP_GROUP_CHAT_MESSAGE = """
👥 <b>Group Chat Mode Enabled!</b>

Bring AI-powered fun to your group:
1️⃣ Add the bot to your group chat
2️⃣ Grant <b>Read Messages</b> (admin role is enough)
3️⃣ Mention me <b>@{bot_username}</b> or reply to any message to get started

Check out the quick tutorial video below for details! 🎥
"""


def split_text_into_chunks(text, chunk_size):
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]

def smart_truncate_message(text, max_length=4096):
    """
    Intelligently truncate a message at natural boundaries with continuation indicator.
    """
    if len(text) <= max_length:
        return text
    
    # Reserve space for continuation indicator
    continuation = "\n\n[... message continues ...]"
    available_length = max_length - len(continuation)
    
    if available_length <= 0:
        return text[:max_length]
    
    # Try to find natural breakpoints in order of preference
    truncated_text = text[:available_length]
    
    # 1. Try to break at last paragraph (double newline)
    last_paragraph = truncated_text.rfind('\n\n')
    if last_paragraph > available_length * 0.7:  # Don't cut too much
        return text[:last_paragraph] + continuation
    
    # 2. Try to break at last sentence
    sentence_endings = ['.', '!', '?']
    last_sentence = -1
    for ending in sentence_endings:
        pos = truncated_text.rfind(ending + ' ')
        if pos > last_sentence:
            last_sentence = pos + 1
    
    if last_sentence > available_length * 0.7:
        return text[:last_sentence] + continuation
    
    # 3. Try to break at last complete word
    last_space = truncated_text.rfind(' ')
    if last_space > available_length * 0.8:
        return text[:last_space] + continuation
    
    # 4. Fallback: hard truncate with indicator, enforce max_length
    fallback = truncated_text + continuation
    return fallback[:max_length]

async def register_user_if_not_exists(update: Update, context: CallbackContext, user: User):
    # register new user; use effective_chat for chat_id
    if not db.check_if_user_exists(user.id):
        chat = getattr(update, 'effective_chat', None)
        chat_id = chat.id if chat else None
        db.add_new_user(
            user.id,
            chat_id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name
        )
        db.start_new_dialog(user.id)

    if db.get_user_attribute(user.id, "current_dialog_id") is None:
        db.start_new_dialog(user.id)

    # ensure per-user semaphore without race
    user_semaphores.setdefault(user.id, asyncio.Semaphore(1))

    if db.get_user_attribute(user.id, "current_model") is None:
        db.set_user_attribute(user.id, "current_model", config.models["available_text_models"][0])

    # back compatibility for n_used_tokens field
    n_used_tokens = db.get_user_attribute(user.id, "n_used_tokens")
    if isinstance(n_used_tokens, int):  # old format
        new_n_used_tokens = {
            "gpt-3.5-turbo-16k": {
                "n_input_tokens": 0,
                "n_output_tokens": n_used_tokens
            }
        }
        db.set_user_attribute(user.id, "n_used_tokens", new_n_used_tokens)

    # voice message transcription
    if db.get_user_attribute(user.id, "n_transcribed_seconds") is None:
        db.set_user_attribute(user.id, "n_transcribed_seconds", 0.0)

    # image generation
    if db.get_user_attribute(user.id, "n_generated_images") is None:
        db.set_user_attribute(user.id, "n_generated_images", 0)
    
    # GPT-Image-1 usage tracking
    if db.get_user_attribute(user.id, "n_gpt_image_1_images") is None:
        db.set_user_attribute(user.id, "n_gpt_image_1_images", 0)

async def is_bot_mentioned(update: Update, context: CallbackContext) -> bool:
    try:
        msg = update.effective_message
        if not msg or not msg.chat:
            return False
        if msg.chat.type == 'private':
            return True
        if msg.text and f"@{context.bot.username}" in msg.text:
            return True
        reply = msg.reply_to_message
        if reply and reply.from_user and reply.from_user.id == context.bot.id:
            return True
    except AttributeError:
        return False
    except Exception:
        logger.exception('Error in is_bot_mentioned')
        return False
    return False

async def start_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    user_id = update.message.from_user.id

    db.set_user_attribute(user_id, "last_interaction", datetime.now())
    db.start_new_dialog(user_id)

    # Warm welcome with feature overview
    reply_text = (
        "👋 <b>Welcome!</b> I’m ChatGPT 🤖, your AI companion for chat, code, images, and more!\n\n"
        "✨ Ready to start? Here are some handy commands:\n"
    )
    reply_text += HELP_MESSAGE

    await update.message.reply_text(reply_text, parse_mode=ParseMode.HTML)
    await show_chat_modes_handle(update, context)

async def help_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())
    # Show refreshed help
    await update.message.reply_text(HELP_MESSAGE, parse_mode=ParseMode.HTML)

async def help_group_chat_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    text = HELP_GROUP_CHAT_MESSAGE.format(bot_username="@" + context.bot.username)

    await update.message.reply_text(text, parse_mode=ParseMode.HTML)
    await update.message.reply_video(
        config.help_group_chat_video_path,
        caption="🎬 How to set me up in group chats"
    )

async def retry_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    dialog_messages = db.get_dialog_messages(user_id, dialog_id=None)
    if len(dialog_messages) == 0:
        await update.message.reply_text("🤷‍♂️ Oops! There's nothing to retry.", parse_mode=ParseMode.HTML)
        return

    last_dialog_message = dialog_messages.pop()
    db.set_dialog_messages(user_id, dialog_messages, dialog_id=None)  # last message was removed from the context

    await message_handle(update, context, message=last_dialog_message["user"], use_new_dialog_timeout=False)

async def message_handle(update: Update, context: CallbackContext, message=None, use_new_dialog_timeout=True):
    # check if bot was mentioned (for group chats)
    if not await is_bot_mentioned(update, context):
        return

    # check if message is edited
    if update.edited_message is not None:
        await edited_message_handle(update, context)
        return

    _message = message or update.message.text

    # remove bot mention (in group chats)
    if update.message.chat.type != "private":
        _message = _message.replace("@" + context.bot.username, "").strip()

    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")

    if chat_mode == "artist":
        await generate_image_handle(update, context, message=message)
        return
    
    if chat_mode == "gpt_image_pro":
        await generate_image_gpt_pro_handle(update, context, message=message)
        return

    current_model = db.get_user_attribute(user_id, "current_model")

    async def message_handle_fn():
        
        # new dialog timeout
        if use_new_dialog_timeout:
            if (datetime.now() - db.get_user_attribute(user_id, "last_interaction")).seconds > config.new_dialog_timeout and len(db.get_dialog_messages(user_id)) > 0:
                db.start_new_dialog(user_id)
                await update.message.reply_text(f"Starting new dialog due to timeout (<b>{config.chat_modes[chat_mode]['name']}</b> mode) ✅", parse_mode=ParseMode.HTML)
        db.set_user_attribute(user_id, "last_interaction", datetime.now())

        # in case of CancelledError
        n_input_tokens, n_output_tokens = 0, 0

        try:
            # send placeholder message to user
            placeholder_message = await update.message.reply_text("...")

            # send typing action
            await update.message.chat.send_action(action="typing")

            if _message is None or len(_message) == 0:
                await update.message.reply_text("😅 You sent <b>empty message</b>. Please, try again!", parse_mode=ParseMode.HTML)
                return

            dialog_messages = db.get_dialog_messages(user_id, dialog_id=None)
            parse_mode = {
                "html": ParseMode.HTML,
                "markdown": ParseMode.MARKDOWN
            }[config.chat_modes[chat_mode]["parse_mode"]]

            chatgpt_instance = openai_utils.ChatGPT(model=current_model)
            if config.enable_message_streaming:
                if chat_mode != "internet_connected_assistant":
                    gen = chatgpt_instance.send_message_stream(_message, dialog_messages=dialog_messages, chat_mode=chat_mode)
                else:
                    gen = chatgpt_instance.send_internetmessage(_message, dialog_messages=dialog_messages, chat_mode="internet_connected_assistant")
            else:
                if chat_mode != "internet_connected_assistant":
                    answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed = await chatgpt_instance.send_message(_message, dialog_messages=dialog_messages, chat_mode=chat_mode)
                    async def fake_gen():
                        yield "finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed

                    gen = fake_gen()
                else:
                    gen = chatgpt_instance.send_internetmessage(_message, dialog_messages=dialog_messages, chat_mode="internet_connected_assistant")

            prev_answer = ""
            async for gen_item in gen:
                status, answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed = gen_item

                answer = smart_truncate_message(answer)  # smart telegram message limit

                # Smart update logic: update more frequently as we approach the limit
                current_length = len(answer)
                length_diff = abs(current_length - len(prev_answer))
                
                # Update thresholds based on message length
                if current_length > 3500:  # Close to limit - update every 50 chars
                    update_threshold = 50
                elif current_length > 2000:  # Medium length - update every 75 chars  
                    update_threshold = 75
                else:  # Short messages - update every 100 chars
                    update_threshold = 100
                
                # Always update when finished or threshold met
                if length_diff < update_threshold and status != "finished":
                    continue

                try:
                    await context.bot.edit_message_text(answer, chat_id=placeholder_message.chat_id, message_id=placeholder_message.message_id, parse_mode=parse_mode)
                except telegram.error.BadRequest as e:
                    if str(e).startswith("Message is not modified"):
                        continue
                    else:
                        await context.bot.edit_message_text(answer, chat_id=placeholder_message.chat_id, message_id=placeholder_message.message_id)

                await asyncio.sleep(0.01)  # wait a bit to avoid flooding

                prev_answer = answer

            # update user data
            new_dialog_message = {"user": _message, "bot": answer, "date": datetime.now()}
            db.set_dialog_messages(
                user_id,
                db.get_dialog_messages(user_id, dialog_id=None) + [new_dialog_message],
                dialog_id=None
            )

            db.update_n_used_tokens(user_id, current_model, n_input_tokens, n_output_tokens)

            #return answer

        except asyncio.CancelledError:
            # note: intermediate token updates only work when enable_message_streaming=True (config.yml)
            db.update_n_used_tokens(user_id, current_model, n_input_tokens, n_output_tokens)
            raise

        except Exception as e:
            error_text = f"Something went wrong during completion. Reason: {e}"
            print(traceback.format_exc())
            logger.error(error_text)
            logger.error(traceback.format_exc())
            await update.message.reply_text(error_text)
            return

        # send message if some messages were removed from the context
        if n_first_dialog_messages_removed > 0:
            if n_first_dialog_messages_removed == 1:
                text = "✍️ <i>Note:</i> Your current dialog is too long, so your <b>first message</b> was removed from the context.\n Send /new command to start new dialog"
            else:
                text = f"✍️ <i>Note:</i> Your current dialog is too long, so <b>{n_first_dialog_messages_removed} first messages</b> were removed from the context.\n Send /new command to start new dialog"
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    async def vision_message_handle_fn():

        # new dialog timeout
        if use_new_dialog_timeout:
            if (datetime.now() - db.get_user_attribute(user_id, "last_interaction")).seconds > config.new_dialog_timeout and len(db.get_dialog_messages(user_id)) > 0:
                db.start_new_dialog(user_id)
                await update.message.reply_text(f"Starting new dialog due to timeout (<b>{config.chat_modes[chat_mode]['name']}</b> mode) ✅", parse_mode=ParseMode.HTML)
        db.set_user_attribute(user_id, "last_interaction", datetime.now())

        buf = None

        if update.message.effective_attachment is not None and len(update.message.effective_attachment) > 0:
            photo = update.message.effective_attachment[-1]
            photo_file = await context.bot.get_file(photo.file_id)

            # store file in memory, not on disk
            buf = io.BytesIO()
            await photo_file.download_to_memory(buf)
            buf.name = "image.jpg"  # file extension is required
            buf.seek(0)  # move cursor to the beginning of the buffer

        # in case of CancelledError
        n_input_tokens, n_output_tokens = 0, 0

        try:
            # send placeholder message to user
            placeholder_message = await update.message.reply_text("...")
            _message = message or update.message.caption or update.message.text

            # send typing action
            await update.message.chat.send_action(action="typing")

            if _message is None or len(_message) == 0:
                await update.message.reply_text("🥲 You sent <b>empty message</b>. Please, try again!", parse_mode=ParseMode.HTML)
                return

            dialog_messages = db.get_dialog_messages(user_id, dialog_id=None)
            parse_mode = {"html": ParseMode.HTML, "markdown": ParseMode.MARKDOWN}[
                config.chat_modes[chat_mode]["parse_mode"]
            ]

            chatgpt_instance = openai_utils.ChatGPT(model=current_model)
            if config.enable_message_streaming:
                gen = chatgpt_instance.send_vision_message_stream(_message, dialog_messages=dialog_messages, chat_mode=chat_mode, image_buffer=buf)
            else:
                (answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed) = await chatgpt_instance.send_vision_message(_message, dialog_messages=dialog_messages, image_buffer=buf, chat_mode=chat_mode)

                async def fake_gen():
                    yield "finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed

                gen = fake_gen()

            prev_answer = ""
            async for gen_item in gen:
                status, answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed = gen_item

                answer = smart_truncate_message(answer)  # smart telegram message limit

                # Smart update logic: update more frequently as we approach the limit
                current_length = len(answer)
                length_diff = abs(current_length - len(prev_answer))
                
                # Update thresholds based on message length
                if current_length > 3500:  # Close to limit - update every 50 chars
                    update_threshold = 50
                elif current_length > 2000:  # Medium length - update every 75 chars  
                    update_threshold = 75
                else:  # Short messages - update every 100 chars
                    update_threshold = 100
                
                # Always update when finished or threshold met
                if length_diff < update_threshold and status != "finished":
                    continue

                try:
                    await context.bot.edit_message_text(answer, chat_id=placeholder_message.chat_id, message_id=placeholder_message.message_id, parse_mode=parse_mode)
                except telegram.error.BadRequest as e:
                    if str(e).startswith("Message is not modified"):
                        continue
                    else:
                        await context.bot.edit_message_text(answer, chat_id=placeholder_message.chat_id, message_id=placeholder_message.message_id)

                await asyncio.sleep(0.01)  # wait a bit to avoid flooding

                prev_answer = answer

            # update user data
            new_dialog_message = {"user": message, "bot": answer, "date": datetime.now()}
            
            db.set_dialog_messages(user_id, db.get_dialog_messages(user_id, dialog_id=None) + [new_dialog_message], dialog_id=None)
            db.update_n_used_tokens(user_id, current_model, n_input_tokens, n_output_tokens)

        except asyncio.CancelledError:
            # note: intermediate token updates only work when enable_message_streaming=True (config.yml)
            db.update_n_used_tokens(user_id, current_model, n_input_tokens, n_output_tokens)
            raise

        except Exception as e:
            error_text = f"Something went wrong during completion. Reason: {e}"
            logger.error(error_text)
            logger.error(traceback.format_exc())
            await update.message.reply_text(error_text)
            return
        
        # send message if some messages were removed from the context
        if n_first_dialog_messages_removed > 0:
            if n_first_dialog_messages_removed == 1:
                text = "✍️ <i>Note:</i> Your current dialog is too long, so your <b>first message</b> was removed from the context.\n Send /new command to start new dialog"
            else:
                text = f"✍️ <i>Note:</i> Your current dialog is too long, so <b>{n_first_dialog_messages_removed} first messages</b> were removed from the context.\n Send /new command to start new dialog"
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    async with user_semaphores[user_id]:
        task = asyncio.create_task(
            vision_message_handle_fn()
            # Always use vision message handling if there's an image attachment
            if (update.message.photo is not None and len(update.message.photo) > 0) or update.message.effective_attachment
            else message_handle_fn()
        )
        user_tasks[user_id] = task

        try:
            await task
        except asyncio.CancelledError:
            await update.message.reply_text("✅ Canceled", parse_mode=ParseMode.HTML)
        else:
            pass
        finally:
            if user_id in user_tasks:
                del user_tasks[user_id]

async def voicemessage_handle(update: Update, context: CallbackContext, message=None, use_new_dialog_timeout=True):
    
    answer = ""  # Variable to store the generated answer
    # check if bot was mentioned (for group chats)
    if not await is_bot_mentioned(update, context):
        return

    # check if message is edited
    if update.edited_message is not None:
        await edited_message_handle(update, context)
        return

    _message = message or update.message.text

    # remove bot mention (in group chats)
    if update.message.chat.type != "private":
        _message = _message.replace("@" + context.bot.username, "").strip()

    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")

    async def voicemessage_handle_fn():
        # new dialog timeout
        if use_new_dialog_timeout:
            if (datetime.now() - db.get_user_attribute(user_id, "last_interaction")).seconds > config.new_dialog_timeout and len(db.get_dialog_messages(user_id)) > 0:
                db.start_new_dialog(user_id)
                await update.message.reply_text(f"Starting new dialog due to timeout (<b>{config.chat_modes[chat_mode]['name']}</b> mode) ✅", parse_mode=ParseMode.HTML)
        db.set_user_attribute(user_id, "last_interaction", datetime.now())

        # in case of CancelledError
        n_input_tokens, n_output_tokens = 0, 0
        current_model = db.get_user_attribute(user_id, "current_model")

        try:
            # send placeholder message to user
            placeholder_message = await update.message.reply_text("...")
            # send typing action
            await update.message.chat.send_action(action="typing")
            if _message is None or len(_message) == 0:
                 await update.message.reply_text("🥲 You sent <b>empty message</b>. Please, try again!", parse_mode=ParseMode.HTML)
                 return
            dialog_messages = db.get_dialog_messages(user_id, dialog_id=None)
            parse_mode = {
                "html": ParseMode.HTML,
                "markdown": ParseMode.MARKDOWN
            }[config.chat_modes[chat_mode]["parse_mode"]]
            chatgpt_instance = openai_utils.ChatGPT(model=current_model)
            if config.enable_message_streaming:
                if chat_mode != "internet_connected_assistant":
                    # Use regular message stream since voice messages don't have images
                    gen = chatgpt_instance.send_message_stream(_message, dialog_messages=dialog_messages, chat_mode=chat_mode)
                else:
                    gen = chatgpt_instance.send_internetmessage(_message, dialog_messages=dialog_messages, chat_mode="internet_connected_assistant")
            else:
                if chat_mode != "internet_connected_assistant":
                    answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed = await chatgpt_instance.send_message(
                        _message,
                        dialog_messages=dialog_messages,
                        chat_mode=chat_mode
                    )
                    async def fake_gen():
                        yield "finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed
                    gen = fake_gen()
                else:
                    gen = chatgpt_instance.send_internetmessage(_message, dialog_messages=dialog_messages, chat_mode="internet_connected_assistant")
            prev_answer = ""
            async for gen_item in gen:
                status, answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed = gen_item
                answer = smart_truncate_message(answer)  # smart telegram message limit
                
                # Smart update logic: update more frequently as we approach the limit
                current_length = len(answer)
                length_diff = abs(current_length - len(prev_answer))
                
                # Update thresholds based on message length
                if current_length > 3500:  # Close to limit - update every 50 chars
                    update_threshold = 50
                elif current_length > 2000:  # Medium length - update every 75 chars  
                    update_threshold = 75
                else:  # Short messages - update every 100 chars
                    update_threshold = 100
                
                # Always update when finished or threshold met
                if length_diff < update_threshold and status != "finished":
                    continue
                    
                try:
                    await context.bot.edit_message_text(answer, chat_id=placeholder_message.chat_id, message_id=placeholder_message.message_id, parse_mode=parse_mode)
                except telegram.error.BadRequest as e:
                    if str(e).startswith("Message is not modified"):
                        continue
                    else:
                        await context.bot.edit_message_text(answer, chat_id=placeholder_message.chat_id, message_id=placeholder_message.message_id)
                await asyncio.sleep(0.01)  # wait a bit to avoid flooding
                prev_answer = answer
            # update user data
            new_dialog_message = {"user": _message, "bot": answer, "date": datetime.now()}
            db.set_dialog_messages(
                user_id,
                db.get_dialog_messages(user_id, dialog_id=None) + [new_dialog_message],
                dialog_id=None
            )
            db.update_n_used_tokens(user_id, current_model, n_input_tokens, n_output_tokens)
            return answer

        except asyncio.CancelledError:
            # note: intermediate token updates only work when enable_message_streaming=True (config.yml)
            db.update_n_used_tokens(user_id, current_model, n_input_tokens, n_output_tokens)
            raise

        except Exception as e:
            error_text = f"Something went wrong during completion. Reason: {e}"
            print(traceback.format_exc())
            logger.error(error_text)
            logger.error(traceback.format_exc())
            await update.message.reply_text(error_text)
            return None

    async with user_semaphores[user_id]:
        task = asyncio.create_task(voicemessage_handle_fn())
        user_tasks[user_id] = task

        try:
            answer = await task
        except asyncio.CancelledError:
            await update.message.reply_text("✅ Canceled", parse_mode=ParseMode.HTML)
        else:
            pass
        finally:
            if user_id in user_tasks:
                del user_tasks[user_id]
    #Return generated answer
    return answer

async def is_previous_message_not_answered_yet(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)

    user_id = update.message.from_user.id
    if user_semaphores[user_id].locked():
        text = "⏳ Please <b>wait</b> for a reply to the previous message\n"
        text += "Or you can /cancel it"
        await update.message.reply_text(text, reply_to_message_id=update.message.id, parse_mode=ParseMode.HTML)
        return True
    else:
        return False

async def voice_message_handle(update: Update, context: CallbackContext):
    logger.debug("Handling voice message")
    
    # check if bot was mentioned (for group chats)
    if not await is_bot_mentioned(update, context):
        logger.debug("Bot not mentioned in group chat")
        return

    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): 
        logger.debug("Previous message not answered yet")
        return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())
    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")

    voice = update.message.voice
    logger.debug(f"Received voice message from user {user_id}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        voice_ogg_path = tmp_dir_path / "voice.ogg"

        # download
        voice_file = await context.bot.get_file(voice.file_id)
        await voice_file.download_to_drive(voice_ogg_path)
        logger.debug(f"Downloaded voice message to {voice_ogg_path}")

        # convert to wav
        voice_wav_path = tmp_dir_path / "voice.wav"
        pydub.AudioSegment.from_file(voice_ogg_path).set_channels(1).set_sample_width(2).set_frame_rate(16000).export(voice_wav_path, format="wav")
        logger.debug(f"Converted voice message to WAV format at {voice_wav_path}")

        transcribed_text = await openai_utils.transcribe_audio(voice_wav_path, openai_utils.config_manager)
        logger.debug(f"Transcribed text: {transcribed_text}")

    text = f"🎤 Detected language <i>{transcribed_text[1]}</i>: <i>{transcribed_text[0]}</i>"
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    # update n_transcribed_seconds
    db.set_user_attribute(user_id, "n_transcribed_seconds", voice.duration + db.get_user_attribute(user_id, "n_transcribed_seconds"))
    message = await voicemessage_handle(update, context, message=transcribed_text[0])

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        tts_output_path = tmp_dir_path / "bot_response.mp3"
        translated_message = await openai_utils.translate_text(message, transcribed_text[1], openai_utils.config_manager)
        logger.debug(f"Translated message: {translated_message}")

        if chat_mode == "rick_sanchez":
            await openai_utils.local_text_to_speech(message, tts_output_path, "ricksanchez", openai_utils.config_manager)
        else:
            await openai_utils.text_to_speech(translated_message, tts_output_path, transcribed_text[1], openai_utils.config_manager)
        logger.debug(f"Generated TTS audio at {tts_output_path}")

        await context.bot.send_audio(update.message.chat_id, audio=tts_output_path.open("rb"))
        logger.debug("Sent TTS audio response to user")

async def generate_image_handle(update: Update, context: CallbackContext, message=None):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    await update.message.chat.send_action(action="upload_photo")

    message = message or update.message.text

    try:
        image_urls = await openai_utils.generate_images(message, n_images=config.return_n_generated_images)
    except Exception as e:
        # Check if it's a content policy violation
        if "safety system" in str(e).lower() or "content policy" in str(e).lower() or "rejected" in str(e).lower():
            text = "🥲 Your request <b>doesn't comply</b> with OpenAI's usage policies.\nWhat did you write there, huh?"
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)
            return
        else:
            raise

    # token usage
    db.set_user_attribute(user_id, "n_generated_images", config.return_n_generated_images + db.get_user_attribute(user_id, "n_generated_images"))

    for i, image_url in enumerate(image_urls):
        await update.message.chat.send_action(action="upload_photo")
        # Handle data URLs for Telegram compatibility
        if image_url.startswith("data:image"):
            # Extract base64 data from data URL
            base64_data = image_url.split(',')[1]
            image_bytes = base64.b64decode(base64_data)
            image_buffer = BytesIO(image_bytes)
            image_buffer.name = "dall_e_3.png"
            await update.message.reply_photo(image_buffer, parse_mode=ParseMode.HTML)
        else:
            # Regular URL
            await update.message.reply_photo(image_url, parse_mode=ParseMode.HTML)

async def generate_image_gpt_pro_handle(update: Update, context: CallbackContext, message=None):
    """Handle image generation and editing with GPT-Image-1 model"""
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    await update.message.chat.send_action(action="upload_photo")

    # Check if user sent an image for editing
    has_image = (update.message.photo is not None and len(update.message.photo) > 0) or update.message.effective_attachment
    message_text = message or update.message.text or update.message.caption

    if not message_text:
        text = "🎨 Please provide a text prompt for image generation or editing!\n\n"
        text += "📝 <b>Examples:</b>\n"
        text += "• <i>A futuristic cityscape at sunset with flying cars</i>\n"
        text += "• <i>Portrait of a wise owl wearing glasses, digital art style</i>\n"
        text += "• Send an image with caption: <i>Remove the background and make it transparent</i>"
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)
        return

    try:
        if has_image:
            # Image editing mode
            await update.message.reply_text("🔧 <b>Image editing mode detected!</b> Processing your image...", parse_mode=ParseMode.HTML)
            
            # Get the image
            if update.message.photo:
                photo = update.message.photo[-1]  # Get highest resolution
            else:
                photo = update.message.effective_attachment
            
            photo_file = await context.bot.get_file(photo.file_id)
            
            # Download image to memory
            import io
            image_buffer = io.BytesIO()
            await photo_file.download_to_memory(image_buffer)
            image_buffer.seek(0)
            
            # Use GPT-Image-1 for editing
            image_urls = await openai_utils.edit_image_gpt_image_1(
                image=image_buffer,
                prompt=message_text,
                n_images=1,
                size="1024x1024"
            )
            
            success_text = f"✨ <b>Image edited successfully!</b>\n📝 Prompt: <i>{message_text}</i>"
            await update.message.reply_text(success_text, parse_mode=ParseMode.HTML)
            
        else:
            # Image generation mode
            await update.message.reply_text("🎨 <b>Generating image with GPT-Image-1...</b>", parse_mode=ParseMode.HTML)
            
            # Use GPT-Image-1 for generation with enhanced parameters
            image_urls = await openai_utils.generate_images_gpt_image_1(
                prompt=message_text,
                n_images=config.return_n_generated_images,
                size="1024x1024"
            )
            
            success_text = f"✨ <b>Image generated successfully!</b>\n📝 Prompt: <i>{message_text}</i>"
            await update.message.reply_text(success_text, parse_mode=ParseMode.HTML)

    except Exception as e:
        # Check if it's a content policy violation
        if "safety system" in str(e).lower() or "content policy" in str(e).lower() or "rejected" in str(e).lower():
            text = "🚫 Your request <b>doesn't comply</b> with OpenAI's usage policies.\n"
            text += "Please try a different prompt that follows content guidelines."
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)
            return
        # Check if GPT-Image-1 is not available
        elif "not available" in str(e).lower() or "not found" in str(e).lower():
            text = "⚠️ GPT-Image-1 model is currently not available.\n"
            text += "Please try again later or switch to the regular Artist mode."
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)
            return
        else:
            # Log the error and show a generic message
            logger.error(f"GPT-Image-1 error: {e}")
            text = f"❌ An error occurred while processing your request: {str(e)[:100]}..."
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)
            return

    # Update usage statistics
    db.set_user_attribute(user_id, "n_gpt_image_1_images", config.return_n_generated_images + db.get_user_attribute(user_id, "n_gpt_image_1_images"))

    # Send the generated/edited images
    for i, image_url in enumerate(image_urls):
        await update.message.chat.send_action(action="upload_photo")
        if has_image:
            caption = f"🔧 Edited with GPT-Image-1 ({i+1}/{len(image_urls)})"
        else:
            caption = f"🎨 Created with GPT-Image-1 ({i+1}/{len(image_urls)})"
        
        # Convert data URL to BytesIO for Telegram compatibility
        if image_url.startswith("data:image"):
            # Extract base64 data from data URL
            base64_data = image_url.split(',')[1]
            image_bytes = base64.b64decode(base64_data)
            image_buffer = BytesIO(image_bytes)
            image_buffer.name = "gpt_image_1.png"
            await update.message.reply_photo(image_buffer, caption=caption, parse_mode=ParseMode.HTML)
        else:
            # Regular URL
            await update.message.reply_photo(image_url, caption=caption, parse_mode=ParseMode.HTML)

async def new_dialog_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    db.start_new_dialog(user_id)
    await update.message.reply_text("🔄 Starting a fresh conversation...", parse_mode=ParseMode.HTML)

    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")
    await update.message.reply_text(f"{config.chat_modes[chat_mode]['welcome_message']}", parse_mode=ParseMode.HTML)

async def cancel_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    if user_id in user_tasks:
        task = user_tasks[user_id]
        task.cancel()
    else:
        await update.message.reply_text("😅 Nothing to cancel right now.", parse_mode=ParseMode.HTML)

def get_chat_mode_menu(page_index: int):
    n_chat_modes_per_page = config.n_chat_modes_per_page
    text = f"🎭 Choose your <b>chat mode</b> ({len(config.chat_modes)} available):"

    # buttons
    chat_mode_keys = list(config.chat_modes.keys())
    page_chat_mode_keys = chat_mode_keys[page_index * n_chat_modes_per_page:(page_index + 1) * n_chat_modes_per_page]

    keyboard = []
    for chat_mode_key in page_chat_mode_keys:
        name = config.chat_modes[chat_mode_key]["name"]
        keyboard.append([InlineKeyboardButton(name, callback_data=f"set_chat_mode|{chat_mode_key}")])

    # pagination
    if len(chat_mode_keys) > n_chat_modes_per_page:
        is_first_page = (page_index == 0)
        is_last_page = ((page_index + 1) * n_chat_modes_per_page >= len(chat_mode_keys))

        if is_first_page:
            keyboard.append([
                InlineKeyboardButton("»", callback_data=f"show_chat_modes|{page_index + 1}")
            ])
        elif is_last_page:
            keyboard.append([
                InlineKeyboardButton("«", callback_data=f"show_chat_modes|{page_index - 1}"),
            ])
        else:
            keyboard.append([
                InlineKeyboardButton("«", callback_data=f"show_chat_modes|{page_index - 1}"),
                InlineKeyboardButton("»", callback_data=f"show_chat_modes|{page_index + 1}")
            ])

    reply_markup = InlineKeyboardMarkup(keyboard)

    return text, reply_markup

async def show_chat_modes_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    text, reply_markup = get_chat_mode_menu(0)
    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)

async def show_chat_modes_callback_handle(update: Update, context: CallbackContext):
     await register_user_if_not_exists(update.callback_query, context, update.callback_query.from_user)
     if await is_previous_message_not_answered_yet(update.callback_query, context): return

     user_id = update.callback_query.from_user.id
     db.set_user_attribute(user_id, "last_interaction", datetime.now())

     query = update.callback_query
     await query.answer()

     page_index = int(query.data.split("|")[1])
     if page_index < 0:
         return

     text, reply_markup = get_chat_mode_menu(page_index)
     try:
         await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
     except telegram.error.BadRequest as e:
         if str(e).startswith("Message is not modified"):
             pass

async def set_chat_mode_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update.callback_query, context, update.callback_query.from_user)
    user_id = update.callback_query.from_user.id

    query = update.callback_query
    await query.answer()

    chat_mode = query.data.split("|")[1]

    db.set_user_attribute(user_id, "current_chat_mode", chat_mode)
    db.start_new_dialog(user_id)

    await context.bot.send_message(
        update.callback_query.message.chat.id,
        f"{config.chat_modes[chat_mode]['welcome_message']}",
        parse_mode=ParseMode.HTML
    )

def get_settings_menu(user_id: int):
    current_model = db.get_user_attribute(user_id, "current_model")
    text = config.models["info"][current_model]["description"]

    text += "\n\n"
    score_dict = config.models["info"][current_model]["scores"]
    for score_key, score_value in score_dict.items():
        text += "🟢" * score_value + "⚪️" * (5 - score_value) + f" – {score_key}\n\n"

    text += "\nSelect <b>model</b>:"

    # buttons to choose models
    buttons = []
    for model_key in config.models["available_text_models"]:
        title = config.models["info"][model_key]["name"]
        if model_key == current_model:
            title = "✅ " + title

        buttons.append(
            InlineKeyboardButton(title, callback_data=f"set_settings|{model_key}")
        )
    reply_markup = InlineKeyboardMarkup([buttons])

    return text, reply_markup

async def settings_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    text, reply_markup = get_settings_menu(user_id)
    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)

async def set_settings_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update.callback_query, context, update.callback_query.from_user)
    user_id = update.callback_query.from_user.id

    query = update.callback_query
    await query.answer()

    _, model_key = query.data.split("|")
    db.set_user_attribute(user_id, "current_model", model_key)
    db.start_new_dialog(user_id)

    text, reply_markup = get_settings_menu(user_id)
    try:
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    except telegram.error.BadRequest as e:
        if str(e).startswith("Message is not modified"):
            pass

async def show_balance_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    # count total usage statistics
    total_n_spent_dollars = 0
    total_n_used_tokens = 0

    n_used_tokens_dict = db.get_user_attribute(user_id, "n_used_tokens")
    n_generated_images = db.get_user_attribute(user_id, "n_generated_images")
    n_gpt_image_1_images = db.get_user_attribute(user_id, "n_gpt_image_1_images")
    n_transcribed_seconds = db.get_user_attribute(user_id, "n_transcribed_seconds")

    details_text = "🏷️ Details:\n"
    for model_key in sorted(n_used_tokens_dict.keys()):
        n_input_tokens, n_output_tokens = n_used_tokens_dict[model_key]["n_input_tokens"], n_used_tokens_dict[model_key]["n_output_tokens"]
        total_n_used_tokens += n_input_tokens + n_output_tokens

        n_input_spent_dollars = config.models["info"][model_key]["price_per_1000_input_tokens"] * (n_input_tokens / 1000)
        n_output_spent_dollars = config.models["info"][model_key]["price_per_1000_output_tokens"] * (n_output_tokens / 1000)
        total_n_spent_dollars += n_input_spent_dollars + n_output_spent_dollars

        details_text += f"- {model_key}: <b>{n_input_spent_dollars + n_output_spent_dollars:.03f}$</b> / <b>{n_input_tokens + n_output_tokens} tokens</b>\n"

    # image generation
    image_generation_n_spent_dollars = config.models["info"]["dall-e-3"]["price_per_1_image"] * n_generated_images
    if n_generated_images != 0:
        details_text += f"- DALL·E 3 (image generation): <b>{image_generation_n_spent_dollars:.03f}$</b> / <b>{n_generated_images} generated images</b>\n"

    total_n_spent_dollars += image_generation_n_spent_dollars

    # GPT-Image-1 generation/editing
    gpt_image_1_n_spent_dollars = config.models["info"]["gpt-image-1"]["price_per_1_image"] * n_gpt_image_1_images
    if n_gpt_image_1_images != 0:
        details_text += f"- GPT-Image-1 (image generation/editing): <b>{gpt_image_1_n_spent_dollars:.03f}$</b> / <b>{n_gpt_image_1_images} generated/edited images</b>\n"

    total_n_spent_dollars += gpt_image_1_n_spent_dollars

    # voice recognition
    voice_recognition_n_spent_dollars = config.models["info"]["whisper"]["price_per_1_min"] * (n_transcribed_seconds / 60)
    if n_transcribed_seconds != 0:
        details_text += f"- Whisper (voice recognition): <b>{voice_recognition_n_spent_dollars:.03f}$</b> / <b>{n_transcribed_seconds:.01f} seconds</b>\n"

    total_n_spent_dollars += voice_recognition_n_spent_dollars


    text = f"You spent <b>{total_n_spent_dollars:.03f}$</b>\n"
    text += f"You used <b>{total_n_used_tokens}</b> tokens\n\n"
    text += details_text

    await update.message.reply_text(text, parse_mode=ParseMode.HTML)

async def edited_message_handle(update: Update, context: CallbackContext):
    if update.edited_message.chat.type == "private":
        text = "🥲 Unfortunately, message <b>editing</b> is not supported"
        await update.edited_message.reply_text(text, parse_mode=ParseMode.HTML)

async def error_handle(update: Update, context: CallbackContext) -> None:
    logger.error(msg="Exception while handling an update:", exc_info=context.error)

    try:
        # collect error message
        tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
        tb_string = "".join(tb_list)
        update_str = update.to_dict() if isinstance(update, Update) else str(update)
        message = (
            f"An exception was raised while handling an update\n"
            f"<pre>update = {html.escape(json.dumps(update_str, indent=2, ensure_ascii=False))}"
            "</pre>\n\n"
            f"<pre>{html.escape(tb_string)}</pre>"
        )

        # split text into multiple messages due to 4096 character limit
        for message_chunk in split_text_into_chunks(message, 4096):
            try:
                await context.bot.send_message(update.effective_chat.id, message_chunk, parse_mode=ParseMode.HTML)
            except telegram.error.BadRequest:
                # answer has invalid characters, so we send it without parse_mode
                await context.bot.send_message(update.effective_chat.id, message_chunk)
    except:
        await context.bot.send_message(update.effective_chat.id, "Some error in error handler")

async def post_init(application: Application):
    await application.bot.set_my_commands([
        BotCommand("/new", "Start new dialog"),
        BotCommand("/mode", "Select chat mode"),
        BotCommand("/retry", "Re-generate response for previous query"),
        BotCommand("/balance", "Show balance"),
        BotCommand("/settings", "Show settings"),
        BotCommand("/help", "Show help message"),
    ])

def run_bot() -> None:
    application = (
        ApplicationBuilder()
        .token(config.telegram_token)
        .concurrent_updates(True)
        .rate_limiter(AIORateLimiter(max_retries=5))
        .http_version("1.1")
        .get_updates_http_version("1.1")
        .post_init(post_init)
        .build()
    )

    # add handlers
    user_filter = filters.ALL
    if len(config.allowed_telegram_usernames) > 0:
        usernames = [x for x in config.allowed_telegram_usernames if isinstance(x, str)]
        user_ids = [x for x in config.allowed_telegram_usernames if isinstance(x, int)]
        user_filter = filters.User(username=usernames) | filters.User(user_id=user_ids)

    application.add_handler(CommandHandler("start", start_handle, filters=user_filter))
    application.add_handler(CommandHandler("help", help_handle, filters=user_filter))
    application.add_handler(CommandHandler("help_group_chat", help_group_chat_handle, filters=user_filter))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & user_filter, message_handle))
    application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND & user_filter, message_handle))
    application.add_handler(CommandHandler("retry", retry_handle, filters=user_filter))
    application.add_handler(CommandHandler("new", new_dialog_handle, filters=user_filter))
    application.add_handler(CommandHandler("cancel", cancel_handle, filters=user_filter))

    if FFMPEG_AVAILABLE:
        application.add_handler(MessageHandler(filters.VOICE & user_filter, voice_message_handle))
    else:
        logger.warning('ffmpeg not found; voice message handling is disabled')

    application.add_handler(CommandHandler("mode", show_chat_modes_handle, filters=user_filter))
    application.add_handler(CallbackQueryHandler(show_chat_modes_callback_handle, pattern="^show_chat_modes"))
    application.add_handler(CallbackQueryHandler(set_chat_mode_handle, pattern="^set_chat_mode"))

    application.add_handler(CommandHandler("settings", settings_handle, filters=user_filter))
    application.add_handler(CallbackQueryHandler(set_settings_handle, pattern="^set_settings"))

    application.add_handler(CommandHandler("balance", show_balance_handle, filters=user_filter))

    application.add_error_handler(error_handle)

    # start the bot
    application.run_polling()

if __name__ == "__main__":
    run_bot()
