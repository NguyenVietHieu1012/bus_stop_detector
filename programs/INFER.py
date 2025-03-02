import cv2
import os
import logging
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class YOLOVideoProcessor:
    """
    Lớp xử lý video sử dụng mô hình YOLO.
    """
    def __init__(self,
                 video_path: str,
                 model: YOLO,
                 output_video_path: str = None,
                 export_folder: str = None,
                 export_video: bool = True,
                 export_frames: bool = False,
                 export_frames_with_annotation: bool = True,
                 conf_threshold: float = 0.7) -> None:
        self.video_path = video_path
        self.model = model
        self.export_video = export_video
        self.export_frames = export_frames
        self.export_frames_with_annotation = export_frames_with_annotation
        self.conf_threshold = conf_threshold
        self.output_video_path = output_video_path
        self.export_folder = export_folder

        # Nếu export frame bật và có folder chỉ định, tạo folder nếu chưa tồn tại
        if self.export_frames and self.export_folder:
            os.makedirs(self.export_folder, exist_ok=True)

        # Mở video
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Không thể mở video: {self.video_path}")

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_counter = 0

        # Nếu export video bật, khởi tạo VideoWriter
        if self.export_video and self.output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(self.output_video_path, fourcc, self.fps,
                                       (self.frame_width, self.frame_height))
        else:
            self.out = None

    def process_video(self) -> None:
        logging.info(f"Đang xử lý video: {self.video_path}")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_counter += 1
            annotated_frame, has_detection = self.process_frame(frame)

            # Xuất frame nếu bật export_frames và có detection
            if self.export_frames and has_detection:
                frame_to_save = annotated_frame if self.export_frames_with_annotation else frame
                if self.export_folder:
                    frame_path = os.path.join(self.export_folder, f"frame_{self.frame_counter:05d}.jpg")
                    cv2.imwrite(frame_path, frame_to_save)

            # Ghi frame đã annotate vào video nếu bật export_video
            if self.export_video and self.out is not None:
                self.out.write(annotated_frame)

            cv2.imshow("YOLO Inference", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cleanup()
        logging.info("Xử lý video hoàn tất.")

    def process_frame(self, frame):
        """
        Thực hiện infer trên 1 frame và vẽ bounding box nếu detection có conf > ngưỡng.
        Trả về tuple: (annotated_frame, has_detection).
        """
        results = self.model(frame)
        annotated_frame = frame.copy()
        has_detection = False

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            bboxes = boxes.xyxy.cpu().numpy()  # mảng numpy với shape [N, 4]
            confs = boxes.conf.cpu().numpy()     # mảng numpy chứa confidence với shape [N]
            for bbox, conf in zip(bboxes, confs):
                if conf > self.conf_threshold:
                    has_detection = True
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{conf:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        return annotated_frame, has_detection

    def cleanup(self) -> None:
        self.cap.release()
        if self.out is not None:
            self.out.release()
        cv2.destroyAllWindows()


def process_folder(model: YOLO,
                   input_folder: str,
                   output_folder: str = None,
                   export_folder: str = None,
                   export_video: bool = True,
                   export_frames: bool = False,
                   export_frames_with_annotation: bool = True,
                   conf_threshold: float = 0.7) -> None:
    # Tạo các folder nếu chưa tồn tại
    if export_video and output_folder:
        os.makedirs(output_folder, exist_ok=True)
    if export_frames and export_folder:
        os.makedirs(export_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(input_folder, filename)
            output_video_path = os.path.join(output_folder, f"infer_{filename}") if export_video and output_folder else None
            logging.info(f"Đang xử lý video: {video_path}")
            processor = YOLOVideoProcessor(
                video_path=video_path,
                model=model,
                output_video_path=output_video_path,
                export_folder=export_folder,
                export_video=export_video,
                export_frames=export_frames,
                export_frames_with_annotation=export_frames_with_annotation,
                conf_threshold=conf_threshold
            )
            processor.process_video()


if __name__ == '__main__':
    # =======================
    # CẤU HÌNH TRỰC TIẾP TẠI ĐÂY
    # =======================
    MODE = "video"  # "video" để xử lý 1 video, "folder" để xử lý toàn bộ video trong folder

    MODEL_PATH = "../models/trained_model_28_2_2025_300epochs.pt"
    CONF_THRESHOLD = 0.7

    EXPORT_VIDEO = True            # True: export video đã annotate, False: không export video
    EXPORT_FRAMES = True           # True: export frame, False: không export frame
    EXPORT_FRAMES_WITH_ANNOTATION = True  # True: xuất frame đã annotate, False: xuất frame gốc

    # Nếu chế độ là video
    if MODE == "video":
        VIDEO_PATH = "../reality_test/20210221_1556_1656/infer/FILE250221-164737-027132-M.MP4"
        OUTPUT_VIDEO_PATH = "infer_250221-164737-027132.mp4"
        EXPORT_FOLDER = "export_frames2"

        model = YOLO(MODEL_PATH)
        processor = YOLOVideoProcessor(
            video_path=VIDEO_PATH,
            model=model,
            output_video_path=OUTPUT_VIDEO_PATH,
            export_folder=EXPORT_FOLDER,
            export_video=EXPORT_VIDEO,
            export_frames=EXPORT_FRAMES,
            export_frames_with_annotation=EXPORT_FRAMES_WITH_ANNOTATION,
            conf_threshold=CONF_THRESHOLD
        )
        processor.process_video()

    # Nếu chế độ là folder
    elif MODE == "folder":
        INPUT_FOLDER = "../reality_test/20210221_1556_1656/infer"
        OUTPUT_FOLDER = "output_videos"
        EXPORT_FOLDER = "export_frames_folder"

        model = YOLO(MODEL_PATH)
        process_folder(
            model=model,
            input_folder=INPUT_FOLDER,
            output_folder=OUTPUT_FOLDER if EXPORT_VIDEO else None,
            export_folder=EXPORT_FOLDER,
            export_video=EXPORT_VIDEO,
            export_frames=EXPORT_FRAMES,
            export_frames_with_annotation=EXPORT_FRAMES_WITH_ANNOTATION,
            conf_threshold=CONF_THRESHOLD
        )
