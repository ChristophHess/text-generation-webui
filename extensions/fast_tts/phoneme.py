"""Uses ctypes and libespeak-ng to get IPA phonemes from text"""
import ctypes
import re
import os
import io
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any, Collection, List, Optional
import logging

if os.name == "nt":
    import msvcrt

# -----------------------------------------------------------------------------


class StreamType(Enum):
    """Type of stream used to record phonemes from eSpeak."""

    MEMORY = "memory"
    NONE = "none"


class Phonemizer:
    """
    Use ctypes and libespeak-ng to get IPA phonemes from text.
    Not thread safe.

    Tries to use libespeak-ng.so or libespeak-ng.so.1 for linux and
    espeak-ng.dll for windows.
    """

    SEEK_SET = 0

    EE_OK = 0

    AUDIO_OUTPUT_SYNCHRONOUS = 0x02
    espeakPHONEMES_IPA = 0x02
    espeakCHARS_AUTO = 0
    espeakSSML = 0x10
    espeakPHONEMES = 0x100

    LANG_SWITCH_FLAG = re.compile(r"\([^)]*\)")

    DEFAULT_CLAUSE_BREAKERS = {",", ";", ":", ".", "!", "?"}

    STRESS_PATTERN = re.compile(r"[ˈˌ]")

    def __init__(
        self,
        default_voice: Optional[str] = None,
        clause_breakers: Optional[Collection[str]] = None,
        stream_type: StreamType = StreamType.MEMORY,
    ):
        self.current_voice: Optional[str] = None
        self.default_voice = default_voice
        self.clause_breakers = clause_breakers or Phonemizer.DEFAULT_CLAUSE_BREAKERS

        self.stream_type = stream_type
        self.libc: Any = None
        self.lib_espeak: Any = None

    def phonemize(
        self,
        text: str,
        voice: Optional[str] = None,
        keep_clause_breakers: bool = False,
        phoneme_separator: Optional[str] = None,
        word_separator: str = " ",
        punctuation_separator: str = "",
        keep_language_flags: bool = False,
        no_stress: bool = False,
        ssml: bool = False,
    ) -> str:
        """
        Return IPA string for text.
        Not thread safe.

        Args:
            text: Text to phonemize
            voice: optional voice (uses self.default_voice if None)
            keep_clause_breakers: True if punctuation symbols should be kept
            phoneme_separator: Separator character between phonemes
            word_separator: Separator string between words (default: space)
            punctuation_separator: Separator string between before punctuation (keep_clause_breakers=True)
            keep_language_flags: True if language switching flags should be kept
            no_stress: True if stress characters should be removed

        Returns:
            ipa - string of IPA phonemes
        """

        # log all arguments
        logging.debug(f"text: {text}")
        logging.debug(f"keep_clause_breakers: {keep_clause_breakers}")
        logging.debug(f"phoneme_separator: {phoneme_separator}")
        logging.debug(f"word_separator: {word_separator}")
        logging.debug(f"punctuation_separator: {punctuation_separator}")
        logging.debug(f"keep_language_flags: {keep_language_flags}")
        logging.debug(f"no_stress: {no_stress}")
        logging.debug(f"ssml: {ssml}")

        if ssml and (self.stream_type == StreamType.NONE):
            raise ValueError("Cannot use SSML without stream")

        self._maybe_init()
        logging.debug(f"stream_type: {self.stream_type}")

        voice = voice or self.default_voice
        logging.debug(f"voice: {voice}")

        if (voice is not None) and (voice != self.current_voice):
            self.current_voice = voice
            voice_bytes = voice.encode("utf-8")
            result = self.lib_espeak.espeak_SetVoiceByName(voice_bytes)
            assert result == Phonemizer.EE_OK, f"Failed to set voice to {voice}"

        missing_breakers = []
        if keep_clause_breakers and self.clause_breakers:
            missing_breakers = [c for c in text if c in self.clause_breakers]

        phoneme_flags = Phonemizer.espeakPHONEMES_IPA
        if phoneme_separator:
            phoneme_flags = phoneme_flags | (ord(phoneme_separator) << 8)

        if self.stream_type == StreamType.MEMORY:
            phoneme_lines = self._phonemize_mem_stream(text, phoneme_separator, ssml)
        elif self.stream_type == StreamType.NONE:
            phoneme_lines = self._phonemize_no_stream(text, phoneme_separator)
        else:
            raise ValueError("Unknown stream type")

        # wait for stream to finish
        if self.stream_type == StreamType.MEMORY:
            self.lib_espeak.espeak_Synchronize()

        # check if we should keep language flags and if phoneme_lines is not none
        if not keep_language_flags and phoneme_lines is not None:
            # Remove language switching flags, e.g. (en)
            phoneme_lines = [
                Phonemizer.LANG_SWITCH_FLAG.sub("", line) for line in phoneme_lines
            ]

        if word_separator != " ":
            # Split/re-join words
            for line_idx in range(len(phoneme_lines)):
                phoneme_lines[line_idx] = word_separator.join(
                    phoneme_lines[line_idx].split()
                )

        # Re-insert clause breakers
        if missing_breakers:
            # pylint: disable=consider-using-enumerate
            for line_idx in range(len(phoneme_lines)):
                if line_idx < len(missing_breakers):
                    phoneme_lines[line_idx] += (
                        punctuation_separator + missing_breakers[line_idx]
                    )

        phonemes_str = word_separator.join(line.strip() for line in phoneme_lines)

        if no_stress:
            # Remove primary/secondary stress markers
            phonemes_str = Phonemizer.STRESS_PATTERN.sub("", phonemes_str)

        # Clean up multiple phoneme separators
        if phoneme_separator:
            phonemes_str = re.sub(
                "[" + re.escape(phoneme_separator) + "]+",
                phoneme_separator,
                phonemes_str,
            )

        return phonemes_str

    def _phonemize_mem_stream(
        self, text: str, phoneme_separator: Optional[str], ssml: bool
    ) -> List[str]:
        # Create a command to run espeak and capture the output
        command = [
            "espeak",
            "-q",  # Quiet mode
            "-x",  # Output phonemes
            "--ipa",  # IPA phonemes
            "-v", "en-us",  # Specify the language variant if needed
        ]
        if ssml:
            command.append("-m")
        if phoneme_separator is not None:
            logging.debug(f"phoneme_separator: {phoneme_separator}")
            command.append(f"--sep", "{phoneme_separator}")
        command.append(text)

        # Run the espeak command and capture its output
        output = subprocess.check_output(command, stderr=subprocess.DEVNULL, universal_newlines=True, encoding="utf-8")

        # Split the output into lines
        phonemes_data = output.strip().splitlines()

        return phonemes_data

    def _phonemize_no_stream(
        self, text: str, phoneme_separator: Optional[str]
    ) -> List[str]:
        phoneme_flags = Phonemizer.espeakPHONEMES_IPA
        if phoneme_separator:
            phoneme_flags = phoneme_flags | (ord(phoneme_separator) << 8)

        text_bytes = text.encode("utf-8")
        text_pointer = ctypes.c_char_p(text_bytes)

        text_flags = Phonemizer.espeakCHARS_AUTO

        phoneme_lines = []
        while text_pointer:
            clause_phonemes = ctypes.c_char_p(
                self.lib_espeak.espeak_TextToPhonemes(
                    ctypes.pointer(text_pointer), text_flags, phoneme_flags,
                )
            )
            if clause_phonemes.value is not None:
                phoneme_lines.append(
                    clause_phonemes.value.decode()  # pylint: disable=no-member
                )
        return phoneme_lines
    
    def open_memstream():
        return io.BytesIO()

    def _maybe_init(self):
        if self.lib_espeak:
            # Already initialized
            return


        # Load dll for windows
        if hasattr(ctypes, 'WinDLL'):
            # self.lib_espeak = ctypes.cdll.LoadLibrary("./libespeak-ng.dll")
            self.lib_espeak = ctypes.WinDLL("C:\Program Files\eSpeak NG\libespeak-ng.dll")
            self.libc = ctypes.cdll.LoadLibrary("msvcrt.dll")

        else:
            try:
                self.lib_espeak = ctypes.cdll.LoadLibrary("libespeak-ng.so")
            except OSError:
                # Try .so.1
                self.lib_espeak = ctypes.cdll.LoadLibrary("libespeak-ng.so.1")

        sample_rate = self.lib_espeak.espeak_Initialize(
            Phonemizer.AUDIO_OUTPUT_SYNCHRONOUS, 0, None, 0
        )
        assert sample_rate > 0, "Failed to initialize libespeak-ng"

        if self.stream_type == StreamType.MEMORY:
            # Initialize libc for memory stream
            if hasattr(ctypes, 'WinDLL'):
                pass
                # self.libc = ctypes.WinDLL("msvcrt.dll")
                # self.stream_type = StreamType.NONE
            else:
                self.libc = ctypes.cdll.LoadLibrary("libc.so.6")
            # self.libc.open_memstream.restype = ctypes.POINTER(ctypes.c_char)
