import os

from PIL import Image
import numpy as np
import typing as tp
from copy import deepcopy


class LSBImageSteganography:
    __BITS_FOR_BITS_COUNT = 3  # bits can be between 1 to 4; = 1 pxl shift
    __BITS_FOR_MSG_LENGHT = 33  # = 11 pxls shift
    __STANDART_MSG_TYPE = "utf-8"

    @staticmethod
    def __load_img(path: str) -> tp.Tuple[bool, tp.Union[np.ndarray | None]]:
        """Image loading function

        Args:
            path (str): path to image

        Returns:
            tp.Union[np.ndarray | None]: None - if cant read the image. np.ndarray - always RGB24 (8 bits per img) data contains
        """

        try:
            img = Image.open(path)
            conv_img = img.convert("RGB", colors=256)
            return True, deepcopy(np.asarray(conv_img))
        except Exception as e:
            return False, None

    @staticmethod
    def __save_img(img_array: np.ndarray, path: str) -> bool:
        if path == "":
            return False

        if not path.endswith(".png"):
            path += ".png"

        try:
            im = Image.fromarray(img_array)
            im.save(path)
            return True
        except Exception as e:
            return False

    @staticmethod
    def __estimate_size(bytes_array: bytes) -> int:
        metadata_bits = (
            LSBImageSteganography.__BITS_FOR_BITS_COUNT
            + LSBImageSteganography.__BITS_FOR_MSG_LENGHT
        )

        msg_len_bits = len(bytes_array) * 8

        total_bits = metadata_bits + msg_len_bits

        return total_bits

    @staticmethod
    def __is_img_and_bits_enough(
        img: np.ndarray, bytes_array: bytes, bits_cnt: int
    ) -> bool:
        required_bits = LSBImageSteganography.__estimate_size(bytes_array)
        image_capacity = (
            3 + (np.prod(img.shape) - 3) * bits_cnt
        )  # cause 3 channels are used for bits_cnt variable
        return image_capacity >= required_bits

    @staticmethod
    def __access_bit(data, num):
        base = int(num // 8)
        shift = int(num % 8)
        return (data[base] >> shift) & 0x1

    def __set_bit_value(value: int, bit_value: int, bit_pos: int) -> int:
        if bit_value == 0:
            zero_mask = 255 & ~(0x1 << bit_pos)
            value = value & zero_mask
        elif bit_value == 1:
            one_mask = 0x1 << bit_pos
            value = value | one_mask
        return value

    @staticmethod
    def __encoding_procedure(
        img: np.ndarray, bytes_array: bytes, bits_cnt: int
    ) -> tp.Tuple[bool, np.ndarray]:
        # write info about bits_cnt using in each channel
        bits_cnt_bits = [
            (bits_cnt >> i) & 0x1
            for i in range(LSBImageSteganography.__BITS_FOR_BITS_COUNT)
        ]
        img[0, 0, 0] = LSBImageSteganography.__set_bit_value(
            img[0, 0, 0], bits_cnt_bits[0], 0
        )
        img[0, 0, 1] = LSBImageSteganography.__set_bit_value(
            img[0, 0, 1], bits_cnt_bits[1], 0
        )
        img[0, 0, 2] = LSBImageSteganography.__set_bit_value(
            img[0, 0, 2], bits_cnt_bits[2], 0
        )

        # write remain info in other part of image
        # - generate array of bits
        msg_len_bits = [
            (len(bytes_array) >> i) & 0x1
            for i in range(LSBImageSteganography.__BITS_FOR_MSG_LENGHT)
        ]

        bits_msg_array = [
            LSBImageSteganography.__access_bit(bytes_array, i)
            for i in range(len(bytes_array) * 8)
        ]

        total_bits = msg_len_bits + bits_msg_array

        # - write array of bits in image
        cur_bit, total_bits_len = 0, len(total_bits)
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                # first pixel is ALWAYS for bit_cnt value!
                if y == 0 and x == 0:
                    continue

                for c in range(img.shape[2]):
                    for bit in range(0, bits_cnt):
                        img[y, x, c] = LSBImageSteganography.__set_bit_value(
                            img[y, x, c], total_bits[cur_bit], bit
                        )

                        cur_bit += 1
                        if cur_bit == total_bits_len:
                            return True, img
        return True, img

    def __get_bit_value_from_img(img: np.ndarray, pos: int, bits_cnt: int) -> int:
        bit_coord = pos % bits_cnt

        channel_coord = ((pos - bit_coord) // bits_cnt) % img.shape[2]

        x_coord = (
            (((pos - bit_coord) // bits_cnt) - channel_coord) // img.shape[2]
        ) % img.shape[1]

        y_coord = (
            (((pos - bit_coord) // bits_cnt) - channel_coord) // img.shape[2] - x_coord
        ) // img.shape[1]

        assert (
            y_coord * img.shape[1] * img.shape[2] * bits_cnt
            + x_coord * img.shape[2] * bits_cnt
            + channel_coord * bits_cnt
            + bit_coord
        ) == pos

        # make shift for 1 left because 1st pixel always store info about bits_cnt
        x_coord += 1
        if x_coord == img.shape[1]:
            x_coord = 0
            y_coord += 1

        return int(
            ((img[y_coord, x_coord, channel_coord] & (0x1 << bit_coord)) >> bit_coord)
        )

    @staticmethod
    def __decoding_procedure(img: np.ndarray) -> tp.Tuple[bool, bytes]:
        bits_cnt_array = [
            img[0, 0, 0] & 0x1,
            img[0, 0, 1] & 0x1,
            img[0, 0, 2] & 0x1,
        ]
        bits_cnt = int(
            bits_cnt_array[0] | (bits_cnt_array[1] << 1) | (bits_cnt_array[2] << 2)
        )
        if not (bits_cnt >= 1 and bits_cnt <= 6):
            return False, b""

        msg_len_bits = []
        for i in range(LSBImageSteganography.__BITS_FOR_MSG_LENGHT):
            msg_len_bits.append(
                LSBImageSteganography.__get_bit_value_from_img(img, i, bits_cnt)
            )
        # print(f"msg_bits: {msg_len_bits}")

        msg_len = 0
        for i in range(0, len(msg_len_bits)):
            msg_len = msg_len | (msg_len_bits[i] << i)

        byte_array_bits = []
        for i in range(msg_len * 8):
            byte_array_bits.append(
                LSBImageSteganography.__get_bit_value_from_img(
                    img, LSBImageSteganography.__BITS_FOR_MSG_LENGHT + i, bits_cnt
                )
            )

        # print(f"byte_bits: {byte_array_bits[0:40]}")
        byte_array = []
        for i in range(msg_len):
            new_byte = byte_array_bits[i * 8]
            for j in range(1, 8):
                new_byte = new_byte | (byte_array_bits[i * 8 + j] << j)
            byte_array.append(new_byte)

        return True, bytes(byte_array)

    @staticmethod
    def encoding_msg(
        img_path: str, msg: str, save_path: str, bits_cnt: int = 1
    ) -> tp.Tuple[bool, str]:
        if not (bits_cnt >= 1 and bits_cnt <= 6):
            return False, "Bad bits per channel arg: must be between 1 and 6"

        res, img_raw = LSBImageSteganography.__load_img(img_path)
        if res == False:
            return False, "Can't read img"

        msg_bytes = bytes(
            msg, encoding=LSBImageSteganography.__STANDART_MSG_TYPE
        )  # for standart working

        if not LSBImageSteganography.__is_img_and_bits_enough(
            img_raw, msg_bytes, bits_cnt
        ):
            return False, "Not enought bits to encode msg into image"

        res, new_img = LSBImageSteganography.__encoding_procedure(
            img_raw, msg_bytes, bits_cnt
        )
        if res == False:
            return False, "Unsespect error during encoding msg"

        res = LSBImageSteganography.__save_img(new_img, save_path)
        if res == False:
            return (False, "Cant save img with steganography")

        return True, "Success"

    @staticmethod
    def decoding_msg(img_path: str) -> tp.Tuple[bool, str]:
        res, img_raw = LSBImageSteganography.__load_img(img_path)
        if res == False:
            return False, "Can't read img"

        res, bytes_res = LSBImageSteganography.__decoding_procedure(img_raw)
        if res == False:
            return False, "Cant decode img"

        msg = bytes_res.decode(LSBImageSteganography.__STANDART_MSG_TYPE)
        return res, msg


def test():
    tests = [
        {
            "path": "./K6/data/test1/",
            "img": "pic1.jpg",
            "msg": "./K6/data/test1/msg1.txt",
            "res": "stegano.png",
        },
        {
            "path": "./K6/data/test2/",
            "img": "pic2.jpg",
            "msg": "./K6/data/test2/msg2.txt",
            "res": "stegano.png",
        },
        {
            "path": "./K6/data/test3/",
            "img": "pic3.png",
            "msg": "./K6/data/test3/msg3.txt",
            "res": "stegano.png",
        },
    ]

    for test in tests:
        img_path = os.path.join(test["path"], test["img"])
        with open(test["msg"], "rt", encoding="utf-8") as tfile:
            text = tfile.read()
            for i in range(1, 7):
                res_path = os.path.join(test["path"], f"{i}_" + test["res"])

                res, msg = LSBImageSteganography.encoding_msg(
                    img_path, text, res_path, i
                )

                if res == False:
                    print("ERROR:", msg)
                    continue

                res, msg = LSBImageSteganography.decoding_msg(res_path)
                print(res, msg[0 : min(200, len(msg))])
                assert msg == text


if __name__ == "__main__":
    test()
