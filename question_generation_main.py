'''Ties question generation together with distractor generation.

Note on cleaning: this module used to strip every punctuation mark from
the document before parsing it.  That destroyed the sentence boundaries
the whole pipeline depends on - "Yum!" lost its exclamation mark and was
re-glued to the following sentence, which is how stems such as
"brand restaurant in Asia to ship supplies from ____ ." and
"as an yum. recognizes its responsibility..." were produced.

Cleaning is now delegated to text_cleaning.clean_document, which removes
PDF layout artefacts while leaving punctuation intact.
'''
from hybrid_assessment_system import HybridAssessmentSystem
from incorrect_answer_generation import IncorrectAnswerGenerator
from question_extraction import QuestionExtractor
from nlp_models import nlp
from text_cleaning import clean_document, register_brand_tokens


class QuestionGeneration:
    '''Generates fill-in-the-blank multiple-choice questions.'''

    def __init__(self, num_questions, num_options):
        self.num_questions = num_questions
        self.num_options = num_options
        self.question_extractor = QuestionExtractor(num_questions)

    def clean_text(self, text):
        '''Strip PDF furniture, keep punctuation, protect brand tokens.'''
        cleaned = clean_document(text)
        register_brand_tokens(nlp, cleaned)
        return cleaned

    def generate_questions_dict(self, document):
        document = self.clean_text(document)
        self.questions_dict = self.question_extractor.get_questions_dict(
            document
        )

        self.incorrect_answer_generator = IncorrectAnswerGenerator(
            document,
            doc=self.question_extractor.doc,
            candidates=self.question_extractor.candidates,
        )

        for question_data in self.questions_dict.values():
            question_data['options'] = (
                self.incorrect_answer_generator.get_all_options_dict(
                    question_data['answer'],
                    self.num_options,
                    question_data.get('answer_label', ''),
                    question_data.get('question', ''),
                )
            )

        return self.questions_dict


class QuestionGeneration_free:
    '''Generates a mixed free-response / multiple-choice assessment.'''

    def __init__(self, num_questions, num_options):
        self.num_questions = num_questions
        self.num_options = num_options
        self.question_extractor = HybridAssessmentSystem(num_questions)

    def clean_text(self, text):
        '''Strip PDF furniture, keep punctuation, protect brand tokens.'''
        cleaned = clean_document(text)
        register_brand_tokens(nlp, cleaned)
        return cleaned

    def generate_questions_dict(self, document):
        document = self.clean_text(document)
        self.questions_dict = self.question_extractor.get_assessment(document)

        self.incorrect_answer_generator = IncorrectAnswerGenerator(
            document,
            doc=self.question_extractor.doc,
            candidates=self.question_extractor.candidates,
        )

        # Only multiple-choice items need options.
        for question_data in self.questions_dict.values():
            if question_data.get('type') != 'Multiple-Choice':
                continue

            question_data['options'] = (
                self.incorrect_answer_generator.get_all_options_dict(
                    question_data['answer'],
                    self.num_options,
                    question_data.get('answer_label', ''),
                    question_data.get('question', ''),
                )
            )

        return self.questions_dict
