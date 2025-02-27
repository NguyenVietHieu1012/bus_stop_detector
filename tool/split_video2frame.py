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
      - output_folder: thư mục lưu frame (tất cả frame của các đoạn cắt sẽ được lưu vào đây)
      - segments: danh sách các đoạn cắt, mỗi đoạn là dict có 'start' và 'end' (định dạng HH:MM:SS)
      - frame_interval_seconds: khoảng thời gian giữa các frame cần lưu
    """
    video_path = video_config['video_path']
    output_folder = video_config['output_folder']
    segments = video_config['segments']
    frame_interval_seconds = video_config.get('frame_interval_seconds', 1.75)

    # Tạo thư mục lưu frame nếu chưa tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Mở video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Không thể mở video: {video_path}")
        return

    # Lấy thông tin fps và tổng số frame
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_interval = int(fps * frame_interval_seconds)

    print(f"Đang xử lý video: {video_path}")
    print(f"fps: {fps}, tổng số frame: {total_frames}")

    # Lấy tên video không có phần mở rộng để đặt tên file frame
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Duyệt qua từng đoạn cắt (tối đa 3 đoạn)
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
        saved_frame_count = 0

        with tqdm(total=(end_frame - start_frame), desc=f"    Processing segment {seg_index}") as pbar:
            while frame_count <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                # Kiểm tra và lưu frame theo khoảng cách frame_interval_seconds
                if (frame_count - start_frame) % frames_per_interval == 0:
                    # Tên file bao gồm tên video, số đoạn và số frame (định dạng 6 chữ số)
                    frame_filename = os.path.join(
                        output_folder,
                        f"{video_name}_seg{seg_index}_frame_{frame_count:06d}.png"
                    )
                    cv2.imwrite(frame_filename, frame)
                    saved_frame_count += 1

                frame_count += 1
                pbar.update(1)

        print(f"    Đã lưu {saved_frame_count} frame cho đoạn {seg_index}.")

    cap.release()
    print(f"Hoàn thành xử lý video: {video_path}\n")


# -------------------------
# CẤU HÌNH CHO NHIỀU VIDEO
# -------------------------
videos = [
    {
        'video_path':
            'C:/Users/Admin/PycharmProjects/GamePro/bus_stop_detector/reality_test/20250217-1423-1526/FILE250217-144316-026846-M.MP4',
        'output_folder': 'saved_frames_daylight',
        'segments': [
            {'start': "00:00:27", 'end': "00:00:31"},
            # Bạn có thể thêm tối đa 3 đoạn cho mỗi video
        ],
        'frame_interval_seconds': 0.2
    },
    {
        'video_path':
            'C:/Users/Admin/PycharmProjects/GamePro/bus_stop_detector/reality_test/20250217-1423-1526/FILE250217-144819-026851-M.MP4',
        'output_folder': 'saved_frames_daylight',
        'segments': [
            {'start': "00:00:15", 'end': "00:00:19"}
        ],
        'frame_interval_seconds': 0.2
    },
    {
        'video_path':
            'C:/Users/Admin/PycharmProjects/GamePro/bus_stop_detector/reality_test/20250217-1423-1526/FILE250217-145120-026854-M.MP4',
        'output_folder': 'saved_frames_daylight',
        'segments': [
            {'start': "00:00:11", 'end': "00:00:15"}
        ],
        'frame_interval_seconds': 0.2
    },
    {
        'video_path':
            'C:/Users/Admin/PycharmProjects/GamePro/bus_stop_detector/reality_test/20250217-1423-1526/FILE250217-145320-026856-M.MP4',
        'output_folder': 'saved_frames_daylight',
        'segments': [
            {'start': "00:00:06", 'end': "00:00:10"},
        ],
        'frame_interval_seconds': 0.2
    },
    {
        'video_path':
            'C:/Users/Admin/PycharmProjects/GamePro/bus_stop_detector/reality_test/20250217-1423-1526/FILE250217-145821-026861-M.MP4',
        'output_folder': 'saved_frames_daylight',
        'segments': [
            {'start': "00:00:46", 'end': "00:00:50"},
        ],
        'frame_interval_seconds': 0.2
    },
    {
        'video_path':
            'C:/Users/Admin/PycharmProjects/GamePro/bus_stop_detector/reality_test/20250217-1423-1526/FILE250217-145922-026862-M.MP4',
        'output_folder': 'saved_frames_daylight',
        'segments': [
            {'start': "00:00:10", 'end': "00:00:14"},
        ],
        'frame_interval_seconds': 0.2
    },
    {
        'video_path':
            'C:/Users/Admin/PycharmProjects/GamePro/bus_stop_detector/reality_test/20250217-1423-1526/FILE250217-150122-026864-M.MP4',
        'output_folder': 'saved_frames_daylight',
        'segments': [
            {'start': "00:00:00", 'end': "00:00:02"},
        ],
        'frame_interval_seconds': 0.2
    },
    {
        'video_path':
            'C:/Users/Admin/PycharmProjects/GamePro/bus_stop_detector/reality_test/20250217-1423-1526/FILE250217-150924-026872-M.MP4',
        'output_folder': 'saved_frames_daylight',
        'segments': [
            {'start': "00:00:11", 'end': "00:00:15"},
            {'start': "00:00:37", 'end': "00:00:51"},
        ],
        'frame_interval_seconds': 0.2
    },
    # Bạn có thể thêm nhiều video khác vào danh sách
]

# -------------------------
# XỬ LÝ TỪNG VIDEO
# -------------------------
for video_config in videos:
    process_video(video_config)