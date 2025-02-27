import cv2
import os
from tqdm import tqdm


def time_str_to_seconds(time_str):
    """
    Chuyển đổi chuỗi thời gian dạng HH:MM:SS thành số giây.
    """
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s


def process_video(video_config):
    """
    Xử lý 1 video theo cấu hình:
      - video_path: đường dẫn video
      - output_folder: thư mục lưu video cắt (mỗi đoạn được lưu thành video riêng)
      - segments: danh sách các đoạn cắt, mỗi đoạn là dict có 'start' và 'end' (định dạng HH:MM:SS)
    """
    video_path = video_config['video_path']
    output_folder = video_config['output_folder']
    segments = video_config['segments']

    # Tạo thư mục lưu video nếu chưa tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Mở video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Không thể mở video: {video_path}")
        return

    # Lấy thông tin fps, chiều rộng, chiều cao và tổng số frame
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Đang xử lý video: {video_path}")
    print(f"fps: {fps}, tổng số frame: {total_frames}")

    # Lấy tên video không có phần mở rộng để đặt tên file video cắt
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Duyệt qua từng đoạn cắt
    for seg_index, seg in enumerate(segments, start=1):
        start_time_str = seg['start']
        end_time_str = seg['end']
        start_seconds = time_str_to_seconds(start_time_str)
        end_seconds = time_str_to_seconds(end_time_str)
        start_frame = int(start_seconds * fps)
        end_frame = int(end_seconds * fps)

        print(f"  Đoạn {seg_index}: từ {start_time_str} đến {end_time_str} "
              f"(frame từ {start_frame} đến {end_frame})")

        # Dịch video đến frame bắt đầu của đoạn
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_count = start_frame

        # Khởi tạo VideoWriter để lưu đoạn video cắt
        output_video_path = os.path.join(output_folder, f"{video_name}_seg{seg_index}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        with tqdm(total=(end_frame - start_frame), desc=f"    Processing segment {seg_index}") as pbar:
            while frame_count <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                # Ghi frame vào VideoWriter để tạo video cắt
                out.write(frame)
                frame_count += 1
                pbar.update(1)

        out.release()
        print(f"    Đã lưu video cho đoạn {seg_index}: {output_video_path}")

    cap.release()
    print(f"Hoàn thành xử lý video: {video_path}\n")


# -------------------------
# CẤU HÌNH CHO NHIỀU VIDEO
# -------------------------
videos = [
    {
        'video_path':
            'C:/Users/Admin/PycharmProjects/GamePro/bus_stop_detector/reality_test/WIN_20250217_14_23_01_Pro.mp4',
        'output_folder': 'WIN_20250217_14_23_01_Pro_videos',  # Folder lưu video cắt
        'segments': [
            {'start': "00:01:51", 'end': "00:01:54"},
            {'start': "00:20:42", 'end': "00:20:46"},
            {'start': "00:36:07", 'end': "00:36:11"},
            {'start': "00:37:15", 'end': "00:37:19"},
            # Bạn có thể thêm tối đa 3 đoạn cho mỗi video
        ]
    },
    # Bạn có thể thêm nhiều video khác vào danh sách
]

# -------------------------
# XỬ LÝ TỪNG VIDEO
# -------------------------
for video_config in videos:
    process_video(video_config)
