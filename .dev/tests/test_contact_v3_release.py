"""Portable expanded data rejects corrupt archives and unsafe members before writing."""
import hashlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from nanoprotein import released_contact_v3 as setup

class ContactV3ArchiveTests(unittest.TestCase):
    def test_download_reuses_only_verified_cache(self):
        payload=b'verified archive'
        with tempfile.TemporaryDirectory() as temp, patch.object(setup, 'BUNDLE_SHA256', hashlib.sha256(payload).hexdigest()):
            root=Path(temp)
            with patch.object(setup.urllib.request, 'urlopen', return_value=io.BytesIO(payload)) as fetch:
                archive=setup.download_bundle(root)
                fetch.assert_called_once_with(setup.BUNDLE_URL, timeout=60)
            with patch.object(setup.urllib.request, 'urlopen') as fetch:
                self.assertEqual(setup.download_bundle(root),archive)
                fetch.assert_not_called()
            archive.write_bytes(b'corrupt old cache')
            with patch.object(setup.urllib.request, 'urlopen', return_value=io.BytesIO(payload)) as fetch:
                setup.download_bundle(root)
                fetch.assert_called_once()
            self.assertEqual(archive.read_bytes(),payload)

    def test_corrupt_download_is_not_installed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with patch.object(setup.urllib.request, 'urlopen', return_value=io.BytesIO(b'corrupt response')), self.assertRaisesRegex(ValueError,'checksum'):
                setup.download_bundle(root)
            self.assertEqual(list((root/'cache').iterdir()),[])

    def test_wrong_checksum_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);archive=root/'bad.tar.gz';archive.write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError,'checksum'):
                setup.extract_bundle(archive,root/'out')
            self.assertFalse((root/'out').exists())

    def test_unsafe_paths_links_duplicates_fail_before_writes(self):
        for name,link,repeat in [('contact-v3/../../escape',False,False),('contact-v3/link',True,False),('contact-v3/file',False,True),('other/file',False,False)]:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);archive=root/'test.tar.gz'
                with tarfile.open(archive,'w:gz') as tar:
                    info=tarfile.TarInfo(name)
                    if link: info.type=tarfile.SYMTYPE;info.linkname='../../escape'
                    tar.addfile(info)
                    if repeat: tar.addfile(info)
                with patch.object(setup,'BUNDLE_SHA256',hashlib.sha256(archive.read_bytes()).hexdigest()),self.assertRaises(ValueError):
                    setup.extract_bundle(archive,root/'out')
                self.assertFalse((root/'out').exists())

    def test_archive_contents_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);archive=root/'test.tar.gz'
            with tarfile.open(archive,'w:gz') as tar:
                data=b'verified payload';info=tarfile.TarInfo('contact-v3/payloads/one.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
            with patch.object(setup,'BUNDLE_SHA256',hashlib.sha256(archive.read_bytes()).hexdigest()):
                setup.extract_bundle(archive,root/'out')
            self.assertEqual((root/'out/contact-v3/payloads/one.json').read_bytes(),data)

if __name__=='__main__': unittest.main()
