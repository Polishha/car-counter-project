import cv2
from ultralytics import YOLO


# ==================================================
# НАСТРОЙКИ
# ==================================================

VIDEO_PATH = "video.mp4"          # Название видео
MODEL_PATH = "yolov8n.pt"         # Модель YOLOv8
OUTPUT_VIDEO = "result.mp4"       # Итоговое видео

# Линия подсчёта.
# Можно ставить любую линию: горизонтальную, вертикальную или диагональную.
# Формат: (x, y)
LINE_POINT_1 = (1050, 300)
LINE_POINT_2 = (1050, 780)

# Направление подсчёта:
# "A_TO_B" — считать только переход с одной стороны линии на другую
# "B_TO_A" — считать только переход в обратную сторону
# "BOTH"   — считать пересечение в любую сторону
COUNT_DIRECTION = "BOTH"

# Классы COCO:
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck
VEHICLE_CLASSES = [2]

CONFIDENCE = 0.20


# ==================================================
# ФУНКЦИИ
# ==================================================

def get_point_side(point, line_point_1, line_point_2):
    x, y = point
    x1, y1 = line_point_1
    x2, y2 = line_point_2

    side = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)

    return side


def check_crossing(previous_center, current_center):
    if previous_center is None:
        return False

    previous_side = get_point_side(previous_center, LINE_POINT_1, LINE_POINT_2)
    current_side = get_point_side(current_center, LINE_POINT_1, LINE_POINT_2)

    # Если точка была с одной стороны, а стала с другой — линия пересечена
    if previous_side * current_side < 0:
        if COUNT_DIRECTION == "BOTH":
            return True

        if COUNT_DIRECTION == "A_TO_B":
            return previous_side > 0 and current_side < 0

        if COUNT_DIRECTION == "B_TO_A":
            return previous_side < 0 and current_side > 0

    return False


def draw_text_with_background(frame, text, position):
    x, y = position

    cv2.rectangle(frame, (x - 10, y - 35), (x + 260, y + 10), (0, 0, 0), -1)

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )


# ==================================================
# ОСНОВНАЯ ЧАСТЬ
# ==================================================

def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Ошибка: не удалось открыть видео.")
        print("Проверь, что файл video.mp4 лежит рядом с main.py")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps == 0:
        fps = 25

    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    car_count = 0

    # ID машин, которые уже были посчитаны
    counted_ids = set()

    # Предыдущий центр каждой машины
    previous_centers = {}

    results = model.track(
        source=VIDEO_PATH,
        stream=True,
        persist=True,
        tracker="botsort.yaml",
        classes=VEHICLE_CLASSES,
        conf=CONFIDENCE,
        verbose=False
    )

    for result in results:
        frame = result.orig_img.copy()

        if result.boxes is not None:
            for box in result.boxes:
                if box.id is None:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                track_id = int(box.id[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())

                if class_id not in VEHICLE_CLASSES:
                    continue

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                current_center = (center_x, center_y)

                previous_center = previous_centers.get(track_id)

                color = (0, 255, 0)

                if track_id not in counted_ids:
                    if check_crossing(previous_center, current_center):
                        car_count += 1
                        counted_ids.add(track_id)
                        color = (0, 0, 255)

                if track_id in counted_ids:
                    color = (0, 0, 255)

                previous_centers[track_id] = current_center

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color,
                    2
                )

                cv2.circle(
                    frame,
                    current_center,
                    5,
                    (255, 0, 0),
                    -1
                )

                label = f"ID:{track_id} car {confidence:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        cv2.line(
            frame,
            LINE_POINT_1,
            LINE_POINT_2,
            (255, 255, 0),
            3
        )

        cv2.putText(
            frame,
            "counting line",
            (LINE_POINT_1[0], LINE_POINT_1[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        draw_text_with_background(
            frame,
            f"Cars: {car_count}",
            (30, 55)
        )

        output.write(frame)

        cv2.imshow("Car Counter", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    output.release()
    cv2.destroyAllWindows()

    print("Готово!")
    print(f"Итоговое видео сохранено: {OUTPUT_VIDEO}")
    print(f"Всего машин посчитано: {car_count}")


if __name__ == "__main__":
    main()