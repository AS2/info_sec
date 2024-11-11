import os

import wave
import numpy as np
import typing as tp
from copy import deepcopy


class LSBAudioSteganography:
    __BITS_FOR_BITS_COUNT = 3  # bits can be between 1 to 4; = 1 pxl shift
    __BITS_FOR_MSG_LENGHT = 33  # = 11 pxls shift
    __STANDART_MSG_TYPE = "utf-8"

    @staticmethod
    def __load_wav(
        path: str,
    ) -> tp.Tuple[bool, tp.Union[bytearray | None], tp.Union[bytearray | None]]:
        """Wav loading function

        Args:
            path (str): path to wav song

        Returns:
            tp.Union[bytearray | None]: None - if cant read the wav. bytearray - array of music
        """

        try:
            song = wave.open(path, mode="rb")
            return True, bytearray(list(song.readframes(song.getnframes()))), song
        except Exception as e:
            return False, None, None

    @staticmethod
    def __save_img(modified_bytes: bytearray, song_descr, path: str) -> bool:
        if path == "":
            return False

        if not path.endswith(".wav"):
            path += ".wav"

        try:
            with wave.open(path, "wb") as fd:
                fd.setparams(song_descr.getparams())
                fd.writeframes(modified_bytes)
            return True
        except Exception as e:
            return False

    @staticmethod
    def __estimate_size(msg_bytes_array: bytes) -> int:
        metadata_bits = (
            LSBAudioSteganography.__BITS_FOR_BITS_COUNT
            + LSBAudioSteganography.__BITS_FOR_MSG_LENGHT
        )

        msg_len_bits = len(msg_bytes_array) * 8

        total_bits = metadata_bits + msg_len_bits

        return total_bits

    @staticmethod
    def __is_song_and_bits_enough(
        song_bytes: bytearray, msg_bytes_array: bytes, bits_cnt: int
    ) -> bool:
        required_bits = LSBAudioSteganography.__estimate_size(msg_bytes_array)
        song_capacity = (
            3 + (len(song_bytes) - 3) * bits_cnt
        )  # cause 3 bytes are used for bits_cnt variable
        return song_capacity >= required_bits

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
        song_byte_array: bytearray, msg_byte_array: bytes, bits_cnt: int
    ) -> tp.Tuple[bool, np.ndarray]:
        # write info about bits_cnt using in song byte
        bits_cnt_bits = [
            (bits_cnt >> i) & 0x1
            for i in range(LSBAudioSteganography.__BITS_FOR_BITS_COUNT)
        ]

        song_byte_array[0] = LSBAudioSteganography.__set_bit_value(
            song_byte_array[0], bits_cnt_bits[0], 0
        )
        song_byte_array[1] = LSBAudioSteganography.__set_bit_value(
            song_byte_array[1], bits_cnt_bits[1], 0
        )
        song_byte_array[2] = LSBAudioSteganography.__set_bit_value(
            song_byte_array[2], bits_cnt_bits[2], 0
        )

        # write remain info in other part of image
        # - generate array of bits
        msg_len_bits = [
            (len(msg_byte_array) >> i) & 0x1
            for i in range(LSBAudioSteganography.__BITS_FOR_MSG_LENGHT)
        ]

        bits_msg_array = [
            LSBAudioSteganography.__access_bit(msg_byte_array, i)
            for i in range(len(msg_byte_array) * 8)
        ]

        total_bits = msg_len_bits + bits_msg_array

        # - write array of bits in image
        cur_bit, total_bits_len = 0, len(total_bits)
        for i in range(3, len(song_byte_array)):
            for bit in range(0, bits_cnt):
                song_byte_array[i] = LSBAudioSteganography.__set_bit_value(
                    song_byte_array[i], total_bits[cur_bit], bit
                )

                cur_bit += 1
                if cur_bit == total_bits_len:
                    return True, song_byte_array

        return True, song_byte_array

    def __get_bit_value_from_song_bytes(
        song_byte_array: bytearray, pos: int, bits_cnt: int
    ) -> int:
        bit_coord = pos % bits_cnt

        i_pos = (pos - bit_coord) // bits_cnt

        assert (i_pos * bits_cnt + bit_coord) == pos

        # make shift for 1 left because 1st pixel always store info about bits_cnt
        i_pos += 3
        return int(((song_byte_array[i_pos] & (0x1 << bit_coord)) >> bit_coord))

    @staticmethod
    def __decoding_procedure(song_byte_array: bytearray) -> tp.Tuple[bool, bytes]:
        bits_cnt_array = [
            song_byte_array[0] & 0x1,
            song_byte_array[1] & 0x1,
            song_byte_array[2] & 0x1,
        ]

        bits_cnt = int(
            bits_cnt_array[0] | (bits_cnt_array[1] << 1) | (bits_cnt_array[2] << 2)
        )

        if not (bits_cnt >= 1 and bits_cnt <= 6):
            return False, b""

        msg_len_bits = []
        for i in range(LSBAudioSteganography.__BITS_FOR_MSG_LENGHT):
            msg_len_bits.append(
                LSBAudioSteganography.__get_bit_value_from_song_bytes(
                    song_byte_array, i, bits_cnt
                )
            )
        # print(f"msg_bits: {msg_len_bits}")

        msg_len = 0
        for i in range(0, len(msg_len_bits)):
            msg_len = msg_len | (msg_len_bits[i] << i)

        byte_array_bits = []
        for i in range(msg_len * 8):
            byte_array_bits.append(
                LSBAudioSteganography.__get_bit_value_from_song_bytes(
                    song_byte_array,
                    LSBAudioSteganography.__BITS_FOR_MSG_LENGHT + i,
                    bits_cnt,
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
        song_path: str, msg: str, save_path: str, bits_cnt: int = 1
    ) -> tp.Tuple[bool, str]:
        if not (bits_cnt >= 1 and bits_cnt <= 6):
            return False, "Bad bits per channel arg: must be between 1 and 6"

        res, song_bytes, song_file_descr = LSBAudioSteganography.__load_wav(song_path)
        if res == False:
            return False, "Can't read img"

        msg_bytes = bytearray(
            bytes(msg, encoding=LSBAudioSteganography.__STANDART_MSG_TYPE)
        )  # for standart working

        if not LSBAudioSteganography.__is_song_and_bits_enough(
            song_bytes, msg_bytes, bits_cnt
        ):
            song_file_descr.close()
            return False, "Not enought bits to encode msg into wav song"

        res, new_img = LSBAudioSteganography.__encoding_procedure(
            song_bytes, msg_bytes, bits_cnt
        )
        if res == False:
            song_file_descr.close()
            return False, "Unsespect error during encoding msg"

        res = LSBAudioSteganography.__save_img(song_bytes, song_file_descr, save_path)
        if res == False:
            song_file_descr.close()
            return (False, "Cant save wav with steganography")

        song_file_descr.close()
        return True, "Success"

    @staticmethod
    def decoding_msg(song_path: str) -> tp.Tuple[bool, str]:
        res, song_bytes, song_file_descr = LSBAudioSteganography.__load_wav(song_path)
        if res == False:
            return False, "Can't read song"

        song_file_descr.close()

        res, bytes_res = LSBAudioSteganography.__decoding_procedure(song_bytes)
        if res == False:
            return False, "Cant decode song"

        msg = bytes_res.decode(LSBAudioSteganography.__STANDART_MSG_TYPE)
        return res, msg


def test():
    tests = [
        {
            "path": "./K7/data/test1/",
            "wav": "mbv-os.wav",
            "msg": "./K7/data/test1/descr.txt",
            "res": "stegano.wav",
        },
        {
            "path": "./K7/data/test2/",
            "wav": "at-x.wav",
            "msg": "./K7/data/test2/descr.txt",
            "res": "stegano.wav",
        },
        {
            "path": "./K7/data/test3/",
            "wav": "af-sh.wav",
            "msg": "./K7/data/test3/descr.txt",
            "res": "stegano.wav",
        },
    ]

    for test in tests:
        wav_path = os.path.join(test["path"], test["wav"])
        with open(test["msg"], "rt", encoding="utf-8") as tfile:
            text = tfile.read()
            for i in range(1, 7):
                res_path = os.path.join(test["path"], f"{i}_" + test["res"])

                res, msg = LSBAudioSteganography.encoding_msg(
                    wav_path, text, res_path, i
                )

                if res == False:
                    print("ERROR:", msg)
                    continue

                res, msg = LSBAudioSteganography.decoding_msg(res_path)
                print(res, msg[0 : min(200, len(msg))])
                assert msg == text


if __name__ == "__main__":
    test()
