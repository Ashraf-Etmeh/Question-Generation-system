'''Document clean-up: layout furniture, punctuation, broken words.

Fast - needs WordNet but none of the spaCy / SBERT models.
'''
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from text_cleaning import _repair_split_words, clean_document  # noqa: E402


class CleanDocumentTest(unittest.TestCase):

    def test_removes_table_of_contents_and_page_numbers(self):
        raw = '\n'.join([
            'Table of Contents',
            'Our Commitment to Integrity........5',
            '12',
            'Employees must report suspected violations of this Code to '
            'the Law Department promptly.',
        ])
        self.assertEqual(
            clean_document(raw),
            'Employees must report suspected violations of this Code to '
            'the Law Department promptly.')

    def test_removes_running_headers(self):
        header = 'YUM! GLOBAL CODE OF CONDUCT'
        pages = [
            f'Rule {n} of this guide is one that every employee is '
            'expected to read and follow.'
            for n in range(1, 5)
        ]
        raw = '\n\n'.join(f'{header}\n\n{page}' for page in pages)
        cleaned = clean_document(raw)
        self.assertNotIn(header, cleaned)
        for page in pages:
            self.assertIn(page, cleaned)

    def test_keeps_punctuation_and_brand_names(self):
        raw = ('A Yum! brand restaurant in Asia ships supplies from the '
               'United States every week.')
        self.assertEqual(clean_document(raw), raw)

    def test_rejoins_words_hyphenated_across_lines(self):
        raw = ('Every employee has a responsi-\nbility to protect the '
               'confidential information of the Company.')
        self.assertIn('responsibility to protect', clean_document(raw))

    def test_reattaches_detached_hyphens_and_punctuation(self):
        raw = ('If you qualify for paid -time-off programs , you must use '
               'those days first .')
        self.assertEqual(
            clean_document(raw),
            'If you qualify for paid-time-off programs, you must use those '
            'days first.')


class RepairSplitWordsTest(unittest.TestCase):
    '''pypdf sometimes splits a word where the PDF positions its glyphs
    separately: "M anagement", "thro ugh", "tak ing".'''

    def test_rejoins_a_word_the_document_uses_elsewhere(self):
        text = ("Harry Caray's M anagement Corporation runs each restaurant. "
                'Restaurant management is demanding.')
        self.assertIn('Management Corporation', _repair_split_words(text))

    def test_checks_every_neighbouring_pair(self):
        # "Wednesday thro" is examined first and must not hide "thro ugh".
        text = ('Other locations run Wednesday thro ugh Tuesday. '
                'Lombard runs Monday through Sunday.')
        self.assertIn('Wednesday through Tuesday', _repair_split_words(text))

    def test_keeps_pairs_of_real_words(self):
        text = ('Overtime is paid for hours worked over time limits. '
                'Log in to the portal, then look into the schedule.')
        self.assertEqual(_repair_split_words(text), text)

    def test_a_fragment_joins_only_one_neighbour(self):
        text = ('A "So n or daughter" means a child. A son or daughter '
                'qualifies, nor does age matter.')
        self.assertIn('"Son or daughter"', _repair_split_words(text))

    def test_prefers_completing_the_following_word(self):
        text = 'Wear employee s hoes at work. All employees need shoes.'
        self.assertIn('Wear employee shoes', _repair_split_words(text))


if __name__ == '__main__':
    unittest.main()
