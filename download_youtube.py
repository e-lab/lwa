from pytube import YouTube

def download_video(url, output_path=None):
    try:
        yt = YouTube(url)
        video = yt.streams.filter(progressive=True, file_extension='mp4').first()
        if output_path is None:
            output_path = "youtube_video/"+video.default_filename
        print(output_path)
        video.download(output_path)
        print("Download completed successfully!")
        return output_path
    except Exception as e:
        print("Error occurred while downloading the video:", e)
        return None

# Example usage:
# Provide the YouTube URL as an argument to the function
# and optionally specify the output path where you want to save the video.
# If no output path is specified, the video will be saved in the current directory.

if __name__ == '__main__':
    # Example 1: Downloading a video and saving it in the current directory
    download_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    # Example 2: Downloading a video and saving it to a specific directory
    # download_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "/path/to/save/video/")
