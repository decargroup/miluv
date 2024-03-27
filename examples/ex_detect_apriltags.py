from miluv.data import DataLoader
import os
import cv2
import apriltag


def main():
    mv = DataLoader(
        "1c",
        cir=False,
        barometer=False,
    )

    data = mv.data

    img_path = os.path.join(
        mv.exp_dir,
        mv.exp_name,
        "ifo002",
        "color",
    )

    imgs = [
        cv2.imread(os.path.join(img_path, img)) for img in os.listdir(img_path)
    ]

    gray_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]

    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)
    results = [detector.detect(gray) for gray in gray_imgs]

    # loop over the AprilTag detection results
    for image, result in zip(imgs, results):
        for r in result:
            # extract the bounding box (x, y)-coordinates for the AprilTag
            # and convert each of the (x, y)-coordinate pairs to integers
            (ptA, ptB, ptC, ptD) = r.corners
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            ptA = (int(ptA[0]), int(ptA[1]))
            # draw the bounding box of the AprilTag detection
            cv2.line(image, ptA, ptB, (0, 255, 0), 2)
            cv2.line(image, ptB, ptC, (0, 255, 0), 2)
            cv2.line(image, ptC, ptD, (0, 255, 0), 2)
            cv2.line(image, ptD, ptA, (0, 255, 0), 2)
            # draw the center (x, y)-coordinates of the AprilTag
            (cX, cY) = (int(r.center[0]), int(r.center[1]))
            cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
            # draw the tag family on the image
            tagFamily = r.tag_family.decode("utf-8")
            cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # show the output image after AprilTag detection
        cv2.imshow("Image", image)
        key = cv2.waitKey(0)

        if key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
