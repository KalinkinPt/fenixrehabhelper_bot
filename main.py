import os
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
import whisper
import pandas as pd
from pydub import AudioSegment
import tempfile
import asyncio
import nest_asyncio

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка модели Whisper
model = whisper.load_model("base")

# Функция для обработки голосовых сообщений
async def voice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        voice = update.message.voice
        file = await context.bot.get_file(voice.file_id)
        with tempfile.NamedTemporaryFile(suffix=".ogg") as ogg_file:
            await file.download_to_drive(ogg_file.name)
            audio = AudioSegment.from_ogg(ogg_file.name)

            with tempfile.NamedTemporaryFile(suffix=".wav") as wav_file:
                audio.export(wav_file.name, format="wav")

                # Распознавание речи
                result = model.transcribe(wav_file.name, language="ru")
                text = result["text"]
                logger.info(f"Распознанный текст: {text}")

                score = extract_berg_score(text)
                if score:
                    df = pd.DataFrame({"Шкала Берга": [score]})
                    with tempfile.NamedTemporaryFile(suffix=".xlsx") as excel_file:
                        df.to_excel(excel_file.name, index=False)
                        await update.message.reply_document(document=excel_file.name, filename="berg_score.xlsx")
                else:
                    await update.message.reply_text("Не удалось распознать баллы по шкале Берга.")
    except Exception as e:
        logger.exception("Ошибка при обработке голосового сообщения:")
        await update.message.reply_text("Произошла ошибка при обработке аудио.")

# Функция для извлечения баллов из текста
def extract_berg_score(text):
    import re
    match = re.search(r"(?:берг[а]?|шкала берга)[^\d]*(\d{1,2})", text.lower())
    if match:
        return int(match.group(1))
    return None

# Основная функция
async def main():
    token = os.getenv("BOT_TOKEN")
    if not token:
        logger.error("Переменная BOT_TOKEN не задана.")
        return

    application = ApplicationBuilder().token(token).build()
    application.add_handler(MessageHandler(filters.VOICE, voice_handler))
    await application.run_polling()

# Запуск с поддержкой nest_asyncio для Railway
if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.get_event_loop().run_until_complete(main())
