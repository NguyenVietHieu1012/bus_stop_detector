import os

# Đường dẫn tới thư mục chứa video
folder_path = './reality_test/20250217-1423-1526'  # Thay đường dẫn này bằng đường dẫn của bạn

# Danh sách các phần mở rộng video cần lọc (có thể thêm bớt tùy ý)
video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')

# Lặp qua các file trong thư mục và in ra tên file nếu có phần mở rộng phù hợp
for file_name in os.listdir(folder_path):
    if file_name.lower().endswith(video_extensions):
        print(file_name)