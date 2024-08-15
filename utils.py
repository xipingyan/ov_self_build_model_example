from pathlib import Path
import yt_dlp # pip install yt_dlp

def download_youtube_video():  
  link='https://youtu.be/kgL5LBM-hFI'
  print(f"Downloading video {link} started")
  
  output_file = Path("downloaded_video.mp4")
  ydl_ops = {"format": "best[ext=mp4]", "outtmpl": output_file.as_posix()}
  with yt_dlp.YoutubeDL(ydl_ops) as ydl:
      ydl.download(link)
  
  print(f"Video saved to {output_file}")
