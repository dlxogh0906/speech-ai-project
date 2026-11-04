"""Protect the distinction between actual-sound transcription and diagnosis."""
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
from mdd.phone_atlas import PHONES, summarize, read_audio, wav_bytes, spectrum


def site(expected, actual, predicted, warning=False):
    return dict(expected=expected, actual=actual, predicted=predicted, warning=warning,
                item='item', speaker='speaker')


class TestPhoneAtlas(unittest.TestCase):
    def test_wrong_pronunciation_correctly_transcribed_is_success_even_without_warning(self):
        rows=[site('ㅅ','ㅅ','ㅅ'),site('ㅆ','ㅅ','ㅅ'),site('ㅆ','ㅅ','ㅆ',True),
              site('ㅅ','ㅆ','ㅅ'),site('ㄹ',None,None,False)]
        actual,expected,conf=summarize(rows,{'ㅅ':4,'ㅆ':5,'ㄹ':6})
        s=next(r for r in actual if r['phone']=='ㅅ')
        self.assertEqual((s['n'],s['correct'],s['normal_n'],s['error_n'],s['error_correct']), (3,2,1,2,1))
        self.assertEqual(s['error_accuracy_pct'],50.)
        self.assertEqual(s['errors_warned_exactly'],0)
        self.assertEqual(sum(r['count'] for r in conf),4) # deletion not a spoken phone
        r=next(r for r in expected if r['phone']=='ㄹ')
        self.assertEqual((r['error_n'],r['exact_transcriptions'],r['exact_warnings']),(1,1,0))

    def test_missing_predictions_remain_in_denominator_and_missing_phones_are_null(self):
        actual,_,conf=summarize([site('ㅈ','ㅈ',None),site('ㅈ','ㅈ','ㅈ')],{'ㅈ':4})
        self.assertEqual(len(actual),51)
        self.assertEqual(len(PHONES),len(set(PHONES)))
        z=next(r for r in actual if r['phone']=='ㅈ')
        self.assertEqual(z['accuracy_pct'],50.)
        absent=next(r for r in actual if r['phone']=='ㅊ')
        self.assertIsNone(absent['accuracy_pct'])
        self.assertEqual({r['predicted']:r['count'] for r in conf},{'∅':1,'ㅈ':1})

    def test_audio_crop_preserves_pcm_and_spectral_scale_is_not_per_example(self):
        x=(np.sin(2*np.pi*1000*np.arange(16000)/16000)*12000).astype('<i2')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.wav'
            with wave.open(str(p),'wb') as f:
                f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(x.tobytes())
            y,sr,offset=read_audio(p,.1,.4)
            np.testing.assert_array_equal(y,x[1600:6400]/32768.)
            self.assertEqual((sr,offset),(16000,.1))
            import io
            with wave.open(io.BytesIO(wav_bytes(p,.1,.4)),'rb') as f:
                self.assertEqual(f.readframes(f.getnframes()),x[1600:6400].tobytes())
        f,t,db1=spectrum(y,sr);_,_,db2=spectrum(y*.5,sr)
        k=int(np.argmin(abs(f-1000)))
        np.testing.assert_allclose(db2[k]-db1[k],20*np.log10(.5),atol=1e-9)


if __name__=='__main__':unittest.main()
