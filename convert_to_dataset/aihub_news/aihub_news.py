import json
import datasets
from datasets.tasks import QuestionAnsweringExtractive

logger = datasets.logging.get_logger(__name__)

class AiHubNewsConfig(datasets.BuilderConfig):
    """BuilderConfig for AiHubNews."""

    def __init__(self, **kwargs):
        """BuilderConfig for AiHubNews.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(AiHubNewsConfig, self).__init__(**kwargs)

_URL = "/opt/ml/final-project-level3-nlp-06/data/news"
_URLS = {
    "train": _URL + "/train/train_original.json",
    "dev": _URL + "/valid/valid_original.json",
}

class AiHubNews(datasets.GeneratorBasedBuilder):
    """AiHubNews: News Data from korean. 30k devided."""

    BUILDER_CONFIGS = [
        AiHubNewsConfig(
            name="AiHubNews",
            version=datasets.Version("1.0.0", ""),
            description="AiHub News dataset 30k",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="AiHub News dataset 30k. many category of data included.",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "category": datasets.Value("string"),
                    "media_type": datasets.Value("string"),
                    "media_sub_type": datasets.Value("string"),
                    "media_name": datasets.Value("string"),
                    "size": datasets.Value("string"),
                    "char_count": datasets.Value("int32"),
                    "publish_date": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "extractive": datasets.Value("string"),
                    "abstractive": datasets.Value("string")
                    # "answers": datasets.features.Sequence(
                    #     {
                    #         "text": datasets.Value("string"),
                    #         "answer_start": datasets.Value("int32"),
                    #     }
                    # ),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://aihub.or.kr/aidata/8054",
            citation="AI HUB",
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        key = 0
        with open(filepath, encoding="utf-8") as f:
            news_data = json.load(f)

            for article in news_data['documents']:
                text = ""
                extractive = ""
                for sentence_data in article['text']:

                    if sentence_data: # 빈배열 필터링
                        text += sentence_data[0]['sentence'] + ' ' # 문장

                        if int(sentence_data[0]['index']) in article['extractive']: #요약문
                            extractive += sentence_data[0]['sentence'] + ' '
                

                yield key, {
                            "id": article['id'],
                            "category": article['category'],
                            "media_type": article['media_type'],
                            "media_sub_type": article['media_sub_type'],
                            "media_name": article['media_name'],
                            "size": article['size'],
                            "char_count": article['char_count'],
                            "publish_date": article['publish_date'],
                            "title": article['title'],
                            "text": text,
                            "extractive": extractive,
                            "abstractive": article['abstractive'][0]
                        }
                key += 1