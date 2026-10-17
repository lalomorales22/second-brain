"""YouTube transcript import: link parsing, caption choice, parsing, and fetching (no network)."""
import io
import json
import sys
import types
import unittest
from unittest.mock import patch

from semantic_gravity_memory import youtube
from semantic_gravity_memory.youtube import Transcript, paragraphs, parse_json3, parse_vtt, pick_track, video_id


def fmt(ext, name='English', url=None, protocol='https'):
    return {'ext': ext, 'name': name, 'url': url or f'https://captions.example/{ext}', 'protocol': protocol}


class VideoIdTests(unittest.TestCase):
    def test_accepts_single_video_links(self):
        for link in ('https://www.youtube.com/watch?v=dQw4w9WgXcQ',
                     'https://youtube.com/watch?v=dQw4w9WgXcQ&list=PL123&t=42s',
                     'www.youtube.com/watch?feature=share&v=dQw4w9WgXcQ',
                     'https://m.youtube.com/watch?v=dQw4w9WgXcQ',
                     'https://music.youtube.com/watch?v=dQw4w9WgXcQ',
                     'https://youtu.be/dQw4w9WgXcQ?si=abc',
                     'youtu.be/dQw4w9WgXcQ',
                     'https://www.youtube.com/shorts/dQw4w9WgXcQ',
                     'https://www.youtube.com/live/dQw4w9WgXcQ?feature=share',
                     'https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ',
                     '  https://www.youtube.com/watch?v=dQw4w9WgXcQ  '):
            self.assertEqual(video_id(link), 'dQw4w9WgXcQ', link)

    def test_rejects_everything_else(self):
        for link in ('', 'not a link', 'https://example.com/watch?v=dQw4w9WgXcQ',
                     'https://www.youtube.com/playlist?list=PL1234567890',
                     'https://www.youtube.com/@somechannel', 'https://youtu.be/short',
                     'https://www.youtube.com/watch?v=dQw4w9WgXcQ_extra',
                     'https://evil-youtube.com/watch?v=dQw4w9WgXcQ',
                     'javascript:alert(1)//youtu.be/dQw4w9WgXcQ'):
            with self.assertRaises(ValueError, msg=link):
                video_id(link)


class ParsingTests(unittest.TestCase):
    ASR_JSON3 = json.dumps({'events': [
        {'tStartMs': 0, 'dDurationMs': 999999, 'id': 1},
        {'tStartMs': 80, 'segs': [{'utf8': 'The'}, {'utf8': ' following', 'tOffsetMs': 240}, {'utf8': ' is a talk.'}]},
        {'tStartMs': 1829, 'aAppend': 1, 'segs': [{'utf8': '\n'}]},
        {'tStartMs': 1839, 'segs': [{'utf8': 'Tom &amp; Jerry\nsay hello'}]},
        {'tStartMs': 65000, 'segs': [{'utf8': 'Later point.'}]},
    ]}).encode()

    def test_json3_keeps_text_and_start_times(self):
        cues = parse_json3(self.ASR_JSON3)
        self.assertEqual(cues, [(0.08, 'The following is a talk.'), (1.839, 'Tom & Jerry say hello'), (65.0, 'Later point.')])

    def test_vtt_rolling_auto_captions_are_not_repeated(self):
        vtt = ('WEBVTT\nKind: captions\nLanguage: en\n\n'
               '00:00:00.080 --> 00:00:01.829 align:start position:0%\n \n'
               'The<00:00:00.320><c> following</c><00:00:00.560><c> is</c>\n\n'
               '00:00:01.829 --> 00:00:01.839 align:start position:0%\nThe following is\n \n\n'
               '00:00:01.839 --> 00:00:04.230 align:start position:0%\nThe following is\n'
               'Andrew<00:00:02.240><c> Scull,</c>\n\n'
               '01:00:04.240 --> 01:00:06.710\nAndrew Scull,\nand mental health.\n')
        self.assertEqual(parse_vtt(vtt), [(0.08, 'The following is'), (1.839, 'Andrew Scull,'),
                                          (3604.24, 'and mental health.')])

    def test_paragraphs_break_at_sentence_ends_after_a_minute(self):
        cues = [(0, 'Hello there.'), (30, 'Still intro.'), (61, 'A minute in.'), (62, 'New topic'),
                (120, 'no punctuation'), (190, 'forced break'), (200, 'end')]
        # 61s is past the minute and the paragraph so far ends a sentence, so it starts a new one;
        # 190s forces a break without punctuation because 120s have passed.
        self.assertEqual(paragraphs(cues), [(0, 'Hello there. Still intro.'),
                                            (61, 'A minute in. New topic no punctuation'),
                                            (190, 'forced break end')])

    def test_consecutive_duplicate_cues_collapse(self):
        self.assertEqual(paragraphs([(0, 'Same'), (1, 'Same'), (2, 'Next')]), [(0, 'Same Next')])


class TrackChoiceTests(unittest.TestCase):
    def test_human_captions_win_over_automatic(self):
        info = {'language': 'en', 'subtitles': {'en': [fmt('vtt'), fmt('json3')]},
                'automatic_captions': {'en-orig': [fmt('json3')]}}
        key, chosen, kind = pick_track(info)
        self.assertEqual((key, chosen['ext'], kind), ('en', 'json3', 'manual'))

    def test_foreign_human_captions_do_not_beat_spoken_language(self):
        info = {'language': 'en', 'subtitles': {'de-DE': [fmt('json3', 'German')], 'live_chat': [fmt('json')]},
                'automatic_captions': {'en-orig': [fmt('json3')], 'de': [fmt('json3', 'German')]}}
        self.assertEqual(pick_track(info)[0::2], ('en-orig', 'auto-generated'))

    def test_spoken_language_asr_beats_machine_translation(self):
        info = {'language': 'es', 'subtitles': {},
                'automatic_captions': {'es-orig': [fmt('json3', 'Spanish')], 'en': [fmt('json3')]}}
        self.assertEqual(pick_track(info)[0::2], ('es-orig', 'auto-generated'))
        self.assertEqual(pick_track(info, 'en')[0::2], ('en', 'auto-translated'))

    def test_vtt_fallback_and_streaming_formats_skipped(self):
        info = {'language': 'en', 'subtitles': {},
                'automatic_captions': {'en': [fmt('json3', protocol='m3u8_native'), fmt('vtt')]}}
        key, chosen, _ = pick_track(info)
        self.assertEqual((key, chosen['ext']), ('en', 'vtt'))

    def test_no_captions_is_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, 'no captions'):
            pick_track({'language': 'en', 'subtitles': {'live_chat': [fmt('json')]}, 'automatic_captions': {}})


class TranscriptDocumentTests(unittest.TestCase):
    def test_document_header_timestamps_and_metadata(self):
        t = Transcript(video_id='dQw4w9WgXcQ', title='AC/DC: Live? <Episode 1>', channel='Some Podcast',
                       duration=3725, upload_date='20260917', language='en', language_name='English',
                       captions='auto-generated', cues=[(0, 'Welcome back.'), (3661.5, 'An hour in.')])
        doc = t.document()
        self.assertTrue(doc.startswith('# AC/DC: Live? <Episode 1>\n\nChannel: Some Podcast\nPublished: 2026-09-17\nLength: 01:02:05\n'))
        self.assertIn('Video: https://www.youtube.com/watch?v=dQw4w9WgXcQ', doc)
        self.assertIn('Transcript: English, auto-generated captions from YouTube', doc)
        self.assertIn('[00:00:00] Welcome back.\n\n[01:01:01] An hour in.\n', doc)
        self.assertEqual(t.filename(), 'AC-DC- Live- -Episode 1 (YouTube).md')
        meta = t.metadata()
        self.assertEqual((meta['origin'], meta['video_id'], meta['published']), ('youtube', 'dQw4w9WgXcQ', '2026-09-17'))
        self.assertIn('mishear', meta['coverage'])
        self.assertNotIn('mishear', Transcript(video_id='dQw4w9WgXcQ', title='x', captions='manual').metadata()['coverage'])

    def test_untitled_video_falls_back_to_id(self):
        self.assertEqual(Transcript(video_id='dQw4w9WgXcQ', title='...').filename(), 'dQw4w9WgXcQ (YouTube).md')


class FakeDownloadError(Exception):
    pass


class FakeYDL:
    info = None
    track = b''
    seen = []

    def __init__(self, options):
        self.options = options

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def extract_info(self, url, download=True):
        FakeYDL.seen.append((url, download, self.options))
        if isinstance(FakeYDL.info, Exception):
            raise FakeYDL.info
        return FakeYDL.info

    def urlopen(self, url):
        return io.BytesIO(FakeYDL.track)


class FetchTests(unittest.TestCase):
    def setUp(self):
        fake = types.ModuleType('yt_dlp')
        fake.YoutubeDL = FakeYDL
        fake.utils = types.SimpleNamespace(DownloadError=FakeDownloadError)
        self.modules = patch.dict(sys.modules, {'yt_dlp': fake})
        self.modules.start()
        FakeYDL.seen = []

    def tearDown(self):
        self.modules.stop()

    def test_fetches_only_captions_and_builds_transcript(self):
        FakeYDL.info = {'id': 'dQw4w9WgXcQ', 'title': 'Episode 42', 'channel': 'The Show', 'duration': 5400,
                        'upload_date': '20260101', 'language': 'en', 'subtitles': {},
                        'automatic_captions': {'en-orig': [fmt('json3')]}}
        FakeYDL.track = ParsingTests.ASR_JSON3
        messages = []
        t = youtube.fetch_transcript('https://youtu.be/dQw4w9WgXcQ?list=PL1', progress=messages.append)
        url, download, options = FakeYDL.seen[0]
        self.assertEqual(url, 'https://www.youtube.com/watch?v=dQw4w9WgXcQ')  # playlist parameters dropped
        self.assertFalse(download)
        self.assertTrue(options['skip_download'] and options['noplaylist'])
        self.assertEqual((t.title, t.channel, t.language, t.captions), ('Episode 42', 'The Show', 'en', 'auto-generated'))
        self.assertEqual(len(t.cues), 3)
        self.assertTrue(any('captions' in m for m in messages))

    def test_youtube_errors_become_readable_messages(self):
        FakeYDL.info = FakeDownloadError('ERROR: [youtube] dQw4w9WgXcQ: Video unavailable. This video is private')
        with self.assertRaisesRegex(ValueError, r'^YouTube: Video unavailable\. This video is private$'):
            youtube.fetch_transcript('https://youtu.be/dQw4w9WgXcQ')

    def test_live_streams_are_refused(self):
        FakeYDL.info = {'id': 'dQw4w9WgXcQ', 'live_status': 'is_live', 'subtitles': {}, 'automatic_captions': {}}
        with self.assertRaisesRegex(ValueError, 'stream'):
            youtube.fetch_transcript('https://youtu.be/dQw4w9WgXcQ')

    def test_missing_yt_dlp_explains_how_to_install(self):
        with patch.dict(sys.modules, {'yt_dlp': None}):
            self.assertFalse(youtube.available())
            with self.assertRaisesRegex(ValueError, r'\.\[youtube\]'):
                youtube.fetch_transcript('https://youtu.be/dQw4w9WgXcQ')


if __name__ == '__main__':
    unittest.main()
