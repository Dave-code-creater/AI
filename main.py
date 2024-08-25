import os
from dotenv import load_dotenv
from discord import Intents, Client, Message
from response import get_response, check_message
import asyncio

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

intents = Intents.default()
intents.message_content = True

class BotClient(Client):
    async def on_ready(self):
        print(f'{self.user} has connected to Discord!')
        
    async def on_message(self, message: Message):
        if message.author == self.user:
            return

        try:
            if message.content.startswith('!'):
                response = get_response(message.content)
                await self.send_long_message(message.channel, response)
            else:
                is_negative = check_message(message.content)
                if is_negative:
                    await self.delete_message(message)
        except Exception as e:
            print(f"Error processing message: {e}")
            await self.on_error('on_message', e)

    async def on_error(self, event, *args, **kwargs):
        import traceback
        error_info = traceback.format_exc()
        print(f"Error in event {event}: {error_info}")

    async def send_long_message(self, channel, content):
        # Split the message into chunks of 2000 characters or less
        for i in range(0, len(content), 2000):
            await channel.send(content[i:i + 2000])

    async def on_disconnect(self):
        print(f'{self.user} has disconnected from Discord!')
        await self.close()

    async def delete_message(self, message):
        try:
            await message.delete()
            print(f"Deleted message from {message.author}: {message.content}")
        except discord.Forbidden as e:
            print(f"Forbidden error when trying to delete message: {e}")
        except discord.HTTPException as e:
            print(f"HTTP exception when trying to delete message: {e}")
def main():
    discord_client = BotClient(intents=intents)
    discord_client.run(TOKEN)

if __name__ == "__main__":
    main()
