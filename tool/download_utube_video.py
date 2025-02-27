import yt_dlp

ydl_opts = {
    'outtmpl': 'C:/Users/Admin/PycharmProjects/GamePro/bus_stop_detector/%(title)s.%(ext)s',
}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download(["https://youtu.be/vtBNKNVgTtE?si=bmJY3xbiGu9YiW13"])