import base64
import io
import json
from typing import Literal, Any
from vertexai.generative_models import (
    Part,
)
from PIL import Image

from gimmick.utils.ich import fetch_images_from_local_fs


def _get_text_part(txt: str, model: str) -> Part | dict[str, Any]:
    if model == "gemini":
        return Part.from_text(txt)
    elif model == "gpt4o":
        return {"type": "text", "text": txt}
    else:
        raise ValueError(f"Invalid model type: {model}")


def _image_to_b64(image: Image.Image) -> str:
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    # image_bytes = self._resize_image_bytes(image_bytes)
    return base64.b64encode(image_bytes.getvalue()).decode("utf-8")


def _get_image_part(
    uri: str, model: str, use_local_b64_images: bool = False
) -> Part | dict[str, Any]:
    if model == "gemini":
        return Part.from_uri(uri, mime_type="image/jpeg")
    elif model == "gpt4o":
        if use_local_b64_images:
            try:
                image = fetch_images_from_local_fs(
                    [uri], return_image_objects=True, verbose=True
                )[0]
                base64_image = _image_to_b64(image)
                uri = f"data:image/jpeg;base64,{base64_image}"
            except Exception as e:
                raise ValueError(f"Failed to load image from {uri}: {e}")
        return {"type": "image_url", "image_url": {"url": uri, "detail": "high"}}
    else:
        raise ValueError(f"Invalid model type: {model}")


GENERATE_ICH_VQA_SYSTEM_PROMPT_V1 = (
    "You are a professional annotator who eagerly creates high-quality exam questions based on an image related to a specific cultural event or facet."
    " Your goal is to formulate questions that test people's prior knowledge about the cultural event or facet, using only the provided image as a reference point."
    " You follow the provided annotation guidelines explicitly to formulate the questions."
)

GENERATE_ICH_VQA_SINGLE_IMAGE_MULTIPLE_CHOICE_PROMPT_TEMPLATE_V1 = """
# Annotation Guidelines
Your task is to create 5 multiple-choice question-answer pairs that meet the following criteria:
1. Questions should not be open-ended.
2. Questions should be specific and focused and only contain a single sentence.
3. Each question should have 4 answer choices, including the correct answer.
4. Answers should contain only a single word or a small number of words.
5. Questions must be answerable by looking at the image and using prior knowledge about the cultural event or facet.
6. Questions must relate to the visual content of the image and simultaneously relate to the cultural event or facet.
7. Do not formulate questions about the provided descriptions alone. Use them only for your own understanding to formulate educated questions.
8. Do not formulate generic questions but only questions which are subject to the respective cultural event or facet.
9. Ensure that the questions do not reflect any knowledge or information from the descriptions that is not visible in the image.
10. Ensure that the questions cannot be answered without looking at the image.

Present your output in JSON format, with each question-answer pair containing the following keys: "question", "choices", "correct".

Important reminders:
- Focus solely on what is visible in the image and what is related to the cultural event or facet 
- Ensure that the questions are answerable from the image combined with prior knowledge of the cultural event or facet
- Ensure that the questions are related to the cultural event or facet and are grounded in the images

Now, please generate 5 question-answer pairs based on the image and the description of the cultural event or facet provided in the following:

# Title of the Cultural Event or Facet
'{LABEL}'

# Description of the Cultural Event or Facet
'{DESCRIPTION}'

# Image of an Aspect of the Cultural Event or Facet
"""

GENERATE_ICH_VQA_SYSTEM_PROMPT_V2 = """
# Your Role

You are a professional annotator specialized in creating educational, insightful, and culturally sensitive question-answer pairs based on provided a intangible cultural heritage item. You will be given the following information related to the item:

- Image: An image representing one aspect of the intangible cultural heritage.
- Countries of Origin: The country or countries where this intangible cultural heritage is recognized.
- Regions of Origin: The country or countries where this intangible cultural heritage is recognized.
- Title: The official title of the intangible cultural heritage item.
- Description: A detailed description of the intangible cultural heritage, including any relevant details.

# Your Task

Your task is it to generate high-quality question-answer pairs in a VQA style to assess the cultural knowledge of the intangible cultural heritage items of state-of-the-art AI models.

Be sure to follow the guidelines provided below to ensure the quality and relevance of the question-answer pairs.

# Question-Answer Pair Guidelines

1.	Generate high-quality question-answer pairs that are directly related to the image and the accompanying information.
    - You can generate up to 10 question-answer pairs for each intangible cultural heritage item. However, ensure that the question-answer pairs must be of high quality and unique.
    - If you are unable to generate 10 high-quality question-answer pairs, you can provide less.
    - Each question must be clear, concise, and directly related to the intangible cultural heritage item, i.e, the image and textual information provided.
    - Ensure that the questions are educational and insightful, encouraging a deeper understanding of the intangible cultural heritage.
    - Ensure that one cannot answer the questions without referring to the image AND knowledge of the intangible cultural heritage. That is, the questions should not be answerable based on the image or the textual information alone.
    - Ensure that the questions do not include words like "likely," "probably,", "possibly", "eventually", etc.
    - Ensure that the questions are referring to the visible content of the image.
    - Ensure that the questions do not contain partial answers or hints to the correct answer.
    - Try to include specific cultural terms, names, or phrases related to the intangible cultural heritage item in the questions.
    - Forumulate the question in a VQA style, where the answer can be directly inferred from the image iff combined with prior knowledge of the intangible cultural heritage item.
    - While generating the question-answer pairs, be aware that ONLY the questions and the image are provided to the AI model. The AI model should be able to answer the questions based on the image and its prior knowledge of the intangible cultural heritage item. This is the single most important aspect of the task.
2.	Ensure diversity by generating questions targeting different aspects of the intangible cultural heritage item such as:
    - Food
    - Drinks
    - Clothing
    - Art
    - Dance
    - Music
    - Instruments
    - Rituals
    - Traditions
    - Festivals
    - Customs
    - Tools
    - Sports
    - Symbols
    - Architecture
    - ... and more
3. Ensure diversity in the categories of questions, including:
    - Identification Questions (e.g., "What is the name of the instrument shown in the image?")
    - Origin Questions (e.g., "Which culture or country does this artifact belong to?")
    - Cultural Significance Questions (e.g., "What cultural or religious significance does this item hold in its native context?")
    - Function or Usage Questions (e.g., "What was this object traditionally used for?")
    - Material and Craftsmanship Questions (e.g., "What material is used to construct this artifact?")
    - Location Questions (e.g., "In which place takes this dance place?")
    - Symbolism Questions (e.g., "What does the color red symbolize in this cultural context?")
    - Historical Questions (e.g., "What historical event is depicted in this image?")
    - Details Questions (e.g., "What formation are the dancers in?")
    - ... and more
4.	Provide accurate and concise answers based solely on the provided information.
    - Each answer must only contain a single word or a multiword expression.
    - Ensure that the answers can be directly found in the provided information.
    - Ensure that the answers are not ambiguous or subjective.
    - Ensure that the answers are directly related to the visible content of the image.
    - Ensure that the answers do not include general, abstract, or non-depictable terms like "Traditional", "Cooperation", "Festival", "Linking", "Gathering", "Solidarity", etc.
    - Try to include specific cultural terms, names, or phrases related to the intangible cultural heritage item in the answers.

# Task Strategy
    
Before generating a question-answer pair, first think step-by-step and analyse the image:

1. What is visible in the image? Generate a highly detailed description of the key elements, objects, or people in the image.
2. How does the visible content relate to the intangible cultural heritage item? Identify the connection between the contents of the image and the intangible cultural heritage item.

Then, think step-by-step about potential questions:

1. What can be asked about the image that is directly related to the visible content and the intangible cultural heritage item?
2. Can a concise and clear answer to the questions be inferred from the image and the provided information?

Finally, think step-by-step before generating the final question-answer pairs:

1. Does the question-answer pair strictly adhere to the guidelines provided above? Percisly check every part of the guidelines and drop the question-answer pair if it does not meet the criteria.
2. What aspect of the intangible cultural heritage item is targeted with the question?
3. What category does the question fall into?

# Output Format

For each question-answer pair, provide the following information in the following format:
```xml
<vqa-task>
    <image-analysis>
        <description>
            <!-- PUT YOUR DETAILED DESCRIPTION OF THE IMAGE HERE -->
        </description>
        <cultural-relatetness>
            <!-- PUT YOUR ANALYSIS OF HOW THE CONTENTS OF THE IMAGE RELATE TO THE INTANGIBLE CULTURAL HERITAGE ITEM HERE -->
        </cultural-relatetness>
    </image-analysis>
    <potential-questions>
        <qa-candidate>
            <question>
                <!-- PUT YOUR QUESTION HERE -->
            </question>
            <answer>
                <!-- PUT YOUR ANSWER HERE -->
            </answer>
        </qa-candidate>
        ...
    </potential-questions>
    <final-qa-pairs>
        <qa-pair>
            <guideline-adherence>
                <!-- DOES YOUR QUESTION-ANSWER PAIR ADHERE TO THE GUIDELINES? YES OR NO -->
            </guideline-adherence>
            <final-result-json>
                <!-- PUT YOUR FINAL RESULT AS JSON HERE -->
                {
                    "question": <insert question here>,
                    "answer": <insert answer here>,
                    "target_aspect": <insert target aspect here>
                    "question_category": <insert question category here>
                }
            </final-result-json>
        </qa-pair>
        ...
    </final-qa-pairs>
</vqa-task>
```

"""


def _get_system_prompt_few_shot_examples(
    model_type: Literal["gemini", "gpt4o"],
    num_examples: int = -1,
    use_local_b64_images: bool = False,
    output_key: str = "Output",
) -> list[Part | dict[str, Any]]:
    examples = [
        {
            "Image": "https://ich.unesco.org/img/photo/thumb/00803-HUG.jpg",
            "Countries of Origin": "Egypt",
            "Regions of Origin": "Arab States",
            "Title": "Al-Sirah Al-Hilaliyyah epic",
            "Description": "This oral poem, also known as the Hilali epic, recounts the saga of the Bani Hilal Bedouin tribe and its migration from the Arabian Peninsula to North Africa in the tenth century. This tribe held sway over a vast territory in central North Africa for more than a century before being annihilated by Moroccan rivals. As one of the major epic poems that developed within the Arabic folk tradition, the Hilali is the only epic still performed in its integral musical form. Moreover, once widespread throughout the Middle East, it has disappeared from everywhere except Egypt. \n\n\nSince the fourteenth century, the Hilali epic has been performed by poets who sing the verses while playing a percussion instrument or a two-string spike fiddle (rabab). Performances take place at weddings, circumcision ceremonies and private gatherings, and may last for days. In the past, practitioners were trained within family circles and performed the epic as their only means of income. These professional poets began their ten-year apprenticeships at the age of five. To this day, students undergo special training to develop memory skills and to master their instruments. Nowadays, they must also learn to inject improvisational commentary in order to render plots more relevant to contemporary audiences. \n\nThe number of performers of the Hilali Epic is dwindling due to competition from contemporary media and to the decreasing number of young people able to commit to the rigorous training process. Pressured by the lucrative Egyptian tourist industry, poets tend to forsake the full Hilali repertory in favour of brief passages performed as part of folklore shows.",
            output_key: [
                {
                    "question": "What instrument are the performers holding in the image related to the Al-Sirah Al-Hilaliyyah epic?",
                    "answer": "Rabab",
                    "target_aspect": "instrument",
                    "question_category": "identification",
                }
            ],
        },
        {
            "Image": "https://ich.unesco.org/img/photo/thumb/06966-HUG.jpg",
            "Countries of Origin": "Croatia",
            "Regions of Origin": "Eastern European States",
            "Title": "Klapa multipart singing of Dalmatia, southern Croatia",
            "Description": "Klapa singing is a multipart singing tradition of the southern Croatian regions of Dalmatia. Multipart singing, a capella homophonic singing, oral tradition and simple music making are its main features. The leader of each singing group is the first tenor, followed by several tenori, baritoni and basi voices. During performances, the singers stand in a tight semicircle. The first tenor starts the singing and is followed by the others. The main aim is to achieve the best possible blend of voices. Technically, klapa singers express their mood by means of open guttural, nasal sotto voce and falsetto singing, usually in high-pitched tessitura. Another feature is the ability to sing freely, without the help of notation. Topics of klapa songs usually deal with love, life situations, and the environment in which they live. Bearers and practitioners are skilled amateurs who inherit the tradition from their predecessors. Their ages vary with many younger people singing with older singers. In ‘traditional klapa’, knowledge is transferred orally. ‘Festival klapa’ is more formally organized with a focus on performance and presentation. In ‘modern klapa’, young singers gain experience by attending performances and listening to recordings. Local communities see klapa singing as a central marker of their musical identity, incorporating respect for diversity, creativity and communication.",
            output_key: [
                {
                    "question": "What formation are the klapa singers standing in during the performance?",
                    "answer": "Semicircle",
                    "target_aspect": "music",
                    "question_category": "details",
                }
            ],
        },
        {
            "Image": "https://ich.unesco.org/img/photo/thumb/07698-HUG.jpg",
            "Countries of Origin": "Spain",
            "Regions of Origin": "Western European and North American States",
            "Title": "Fiesta of the patios in Cordova",
            "Description": "For twelve days at the beginning of May, the city of Cordova celebrates the Fiesta of the Patios. The patio houses are communal, family or multi-family dwellings or sets of individual houses with a shared patio, located in the city’s historical quarter. This characteristic cultural space boasts an abundant array of plants, and during the fiesta inhabitants freely welcome all visitors to admire their beauty and the skill involved in their creation. The patios also host traditional singing, flamenco guitar playing and dancing. Ancestral practices of sustainable communal coexistence are shared with people who visit through expressions of affection and shared food and drink. The fiesta is perceived as an integral part of this city’s cultural heritage, imbuing it with a strong sense of identity and continuity. It requires the selfless cooperation of many people from all age groups, social strata and backgrounds, promoting and encouraging teamwork and contributing to local harmony and conviviality. It is guided by secular traditions, knowledge and skills, which take form in the luxuriant, floral, chromatic, acoustic, aromatic and compositional creativity of each patio – an expression of the symbolism and traditions of Cordovan community, and especially the residents who dwell in these patio houses.",
            output_key: [
                {
                    "question": "What type of community space is depicted in the image during the Fiesta of the Patios?",
                    "answer": "Patio",
                    "target_aspect": "festivals",
                    "question_category": "location",
                }
            ],
        },
        {
            "Image": "https://ich.unesco.org/img/photo/thumb/09849-HUG.jpg",
            "Countries of Origin": "Kazakhstan",
            "Regions of Origin": "Asian and Pacific States",
            "Title": "Kuresi in Kazakhstan",
            "Description": "Kuresi in Kazakhstan is a type of wrestling that requires players to battle it out on foot, the objective being to get the opponent’s shoulders on the ground. It is a traditional practice where trainers would coach young boys who would then take part in local contests. These days, kuresi in Kazakhstan is a national sport practised by men and women, up to professional level. International competitions also take place, such as the annual tournament the Kazakhstan Barysy, broadcast in more than 100 countries. Transmission of kuresi in Kazakhstan occurs in sports clubs, which may also be affiliated to schools, as well as via master classes run by experienced kuresi wrestlers. The minimum age of learners can be as young as 10 and no restrictions apply concerning the background of participants. The sport of kuresi also has a place in traditional folklore in Kazakhstan. The wrestlers, known as Baluans, have been regarded as strong and courageous and depicted as such in epics, poetry and literature. The practice of kuresi teaches younger generations in Kazakhstan to respect their history and culture, and aim to be like the heroic Baluans. It also helps to build tolerance, goodwill and solidarity amongst communities.",
            output_key: [
                {
                    "question": "What traditional Kazakh sport is depicted in the image?",
                    "answer": "Kuresi",
                    "target_aspect": "sports",
                    "question_category": "identification",
                }
            ],
        },
        {
            "Image": "https://ich.unesco.org/img/photo/thumb/10416-HUG.jpg",
            "Countries of Origin": "Morocco",
            "Regions of Origin": "Arab States",
            "Title": "Gnawa",
            "Description": "Gnawa refers to a set of musical events, performances, fraternal practices and therapeutic rituals mixing the secular with the sacred. Gnawa is first and foremost a Sufi brotherhood music combined with lyrics with a generally religious content, invoking ancestors and spirits. Originally practised by groups and individuals from slavery and the slave trade dating back to at least the 16th century, Gnawa culture is now considered as part of Morocco’s multifaceted culture and identity. The Gnawa, especially in the city, practise a therapeutic possession ritual through all-night rhythm and trance ceremonies combining ancestral African practices, Arab-Muslim influences and native Berber cultural performances. The Gnawa in rural areas organize communal meals offered to marabout saints. Some Gnawa in urban areas use a stringed musical instrument and castanets, while those in rural areas use large drums and castanets. Colourful, embroidered costumes are worn in the city, while white attire with accessories characterize rural practices. The number of fraternal groups and master musicians is constantly growing in Morocco’s villages and major cities, and Gnawa groups – organized into associations – hold local, regional, national and international festivals year-round. This allows young people to learn about both the lyrics and musical instruments as well as practices and rituals related to Gnawa culture generally.",
            output_key: [
                {
                    "question": "Which cultural element is associated with the musicians in the image?",
                    "answer": "Gnawa",
                    "target_aspect": "rituals",
                    "question_category": "identification",
                }
            ],
        },
        {
            "Image": "https://ich.unesco.org/img/photo/thumb/12630-HUG.jpg",
            "Countries of Origin": "France, Italy, Switzerland",
            "Regions of Origin": "Western European and North American States",
            "Title": "Alpinism",
            "Description": "Alpinism is the art of climbing up summits and walls in high mountains, in all seasons, in rocky or icy terrain. It involves physical, technical and intellectual abilities, using appropriate techniques, equipment and highly specific tools such as axes and crampons. Alpinism is a traditional, physical practice characterized by a shared culture made up of knowledge of the high-mountain environment, the history of the practice and associated values, and specific skills. Knowledge about the natural environment, changing weather conditions, and natural hazards is also essential. Alpinism is also based on aesthetic aspects: alpinists strive for elegant climbing motions, contemplation of the landscape, and harmony with the natural environment. The practice mobilizes ethical principles based on each individual’s commitment, such as leaving no lasting traces behind, and assuming the duty to provide assistance among practitioners. Another essential part of the alpinist mindset is the sense of team spirit, as represented by the rope connecting the alpinists. Most community members belong to alpine clubs, which spread alpine practices worldwide. The clubs organize group outings, disseminate practical information and contribute to various publications, acting as a driving force for alpinist culture. Since the 20th century, alpine clubs in all three countries have cultivated relationships through frequent bilateral or trilateral meetings at various levels.",
            output_key: [
                {
                    "question": "What activity is being performed by the individuals in the image?",
                    "answer": "Alpinism",
                    "target_aspect": "traditions",
                    "question_category": "identification",
                }
            ],
        },
        {
            "Image": "https://ich.unesco.org/img/photo/thumb/14714-HUG.jpg",
            "Countries of Origin": "Türkiye",
            "Regions of Origin": "Western European and North American States",
            "Title": "Hüsn-i Hat, traditional calligraphy in Islamic art in Turkey",
            "Description": "The Hüsn-i hat is the centuries-old practice of writing letters of Arabic origin in a measured and proportional manner while taking into consideration certain aesthetic values. Traditional tools include a specific type of paper glazed with organic substances, a reed pen, pen knives, a special slab for trimming the reed pen, an inkwell, soot ink and a pen case. Many calligraphers, or hattats, make their own tools, and they play an important role in the transmission of the Hüsn-i hat tradition, passing on their knowledge, craft skills and values through apprenticeships. The Hüsn-i hat can be written on paper or leather. It may also be applied on stone, marble, glass and wood, among others. There are many different styles of Hüsn-i hat, and the practice was traditionally used to write the Koran, hadiths (statements of the Prophet Muhammad) and poetry, as well as for State correspondence, such as imperial edicts and warrants, and on religious and public buildings. In Islam, Hüsn-i hat is seen as a means not only of writing ideas, but of depicting them visually. To this day, Hüsn-i hat is still used in sacred and literary works and on mosques, Turkish baths and temples.",
            output_key: [
                {
                    "question": "Which cultural art form does this image depict?",
                    "answer": "Hüsn-i Hat",
                    "target_aspect": "art",
                    "question_category": "identification",
                }
            ],
        },
        {
            "Image": "https://ich.unesco.org/img/photo/thumb/12141-HUG.jpg",
            "Countries of Origin": "Norway",
            "Regions of Origin": "Western European and North American States",
            "Title": "Practice of traditional music and dance in Setesdal, playing, dancing and singing (stev/stevjing)",
            "Description": "In the practice of traditional music and dance in Setesdal, playing, dancing and singing (stev/stevjing), traditional dance and music belong together, interwoven in the social context. The melodies are named after the ‘gangar’ dance and are mostly dance tunes; the melodies of the ‘stev’ songs can be played on instruments, and the lyrics often describe the dancing or playing of the practitioners. The ‘stev’ are often performed in the intervals between the dancing and playing, and are sung solo or by two or more singers in a dialogue with each other called ‘stevjing’. The lyrics are four-line verses telling a story. The dance is practised either by solo couples or by multiple couples in a clockwise circle with a change of dance partners and can be performed in either a modest way or wildly and vigorously. The music is performed on the ‘Hardanger’ fiddle, Norway’s national instrument, and the jaw harp. Setesdal can be traced back to the 18th century, and has enjoyed continuous transmission. It is constantly evolving, with new song texts being made for traditional ‘stev’ tunes, and new tunes composed. The traditional form of transmission – through social gatherings or from adult experts to younger generations – remains the main method of learning the element.",
            output_key: [
                {
                    "question": "In which country is the practice involving the instrument in the image predominantly found?",
                    "answer": "Norway",
                    "target_aspect": "music",
                    "question_category": "location",
                }
            ],
        },
        {
            "Image": "https://ich.unesco.org/img/photo/thumb/08420-HUG.jpg",
            "Countries of Origin": "Ukraine",
            "Regions of Origin": "Eastern European States",
            "Title": "Petrykivka decorative painting as a phenomenon of the Ukrainian ornamental folk art",
            "Description": "The people of the village of Petrykivka decorate their living quarters, household belongings and musical instruments with a style of ornamental painting that is characterized by fantastic flowers and other natural elements, based on careful observation of the local flora and fauna. This art is rich in symbolism: the rooster stands for fire and spiritual awakening, while birds represent light, harmony and happiness. In folk belief, the paintings protect people from sorrow and evil. Local people, and in particular women of all ages, are involved in this folk art tradition. Every family has at least one practitioner, making decorative painting an integral part of daily existence in the community. The painting traditions, including the symbolism of the ornamental elements, are transferred, renewed and enhanced from one generation to another. Local schools at all levels, from pre-school to college, teach the fundamentals of Petrykivka decorative painting, with all children given the opportunity to study it. The community willingly teaches its skills and know-how to anyone who shows an interest. The tradition of decorative and applied arts contributes to the renewal of historical and spiritual memory and defines the identity of the entire community.",
            output_key: [
                {
                    "question": "Which type of traditional art is represented in the decorative patterns seen in the image?",
                    "answer": "Petrykivka",
                    "target_aspect": "art",
                    "question_category": "craftmenship",
                }
            ],
        },
    ]

    example_parts = []
    if num_examples > 0 or num_examples == -1:
        example_parts.append(_get_text_part("# Examples \n\n", model_type))
        example_parts.append(
            _get_text_part(
                "Here are some examples of intangible cultural heritage items with their respective images, titles, countries of origin, regions of origin, and descriptions followed by the generated question-answer pairs.\n",
                model_type,
            )
        )
        example_parts.append(
            _get_text_part(
                "Note that we only provide examples with a single question-answer pair for illustrative purposes. You are expected to generate up to 10 question-answer pairs for the provided cultural heritage item.\n\n",
                model_type,
            )
        )

        for i, ex in enumerate(examples):
            parts = []
            parts.append(
                _get_text_part(
                    f"## Intangible Cultural Heritage Item (Example {i}):\n\n",
                    model_type,
                )
            )
            for k, v in ex.items():
                parts.append(_get_text_part(f"### {k}:\n", model_type))
                if k == "Image":
                    value_part = _get_image_part(v, model_type, use_local_b64_images)
                elif k == output_key:
                    value_part = _get_text_part(json.dumps(v, indent=2), model_type)
                else:
                    value_part = _get_text_part(v, model_type)
                parts.append(_get_text_part("\n", model_type))
                parts.append(value_part)
            parts.append(_get_text_part("\n\n", model_type))
            example_parts.extend(parts)
            if i == num_examples - 1:
                break

    example_parts.append(
        _get_text_part(
            "\n\n Now, generate high-quality question-answer pairs based on the provided intangible cultural heritage Item below.\n\n",
            model_type,
        )
    )

    return example_parts


def get_single_image_open_question_prompt(
    image_name: str,
    title: str,
    countries: str,
    regions: str,
    description: str,
    model_type: Literal["gemini", "gpt4o"],
    output_key: str = "Output",
    num_few_shot_examples: int = -1,
    use_local_b64_images: bool = False,
) -> list[Part] | dict[str, Any]:
    few_shot_parts = _get_system_prompt_few_shot_examples(
        num_examples=num_few_shot_examples,
        model_type=model_type,
        use_local_b64_images=use_local_b64_images,
        output_key=output_key,
    )

    sample_parts = [
        _get_text_part("# Intangible Cultural Heritage Item\n\n", model_type),
        _get_text_part("### Image:\n", model_type),
        _get_image_part(image_name, model_type, use_local_b64_images),
        _get_text_part("\n", model_type),
        _get_text_part("### Countries of Origin:\n", model_type),
        _get_text_part(countries, model_type),
        _get_text_part("\n", model_type),
        _get_text_part("### Regions of Origin:\n", model_type),
        _get_text_part(regions, model_type),
        _get_text_part("\n", model_type),
        _get_text_part("### Title:\n", model_type),
        _get_text_part(title, model_type),
        _get_text_part("\n", model_type),
        _get_text_part("### Description:\n", model_type),
        _get_text_part(description, model_type),
        _get_text_part("\n", model_type),
    ]

    all_parts = few_shot_parts + sample_parts

    if model_type == "gpt4o":
        return {"role": "user", "content": all_parts}

    return all_parts


GENERATE_ICH_VQA_SYSTEM_PROMPT_V3 = """
# Your Role

You are a professional annotator specialized in creating VQA samples based on provided a intangible cultural heritage(ICH) item. You will be given the following information related to the item:

- Image: An image representing one aspect of the ICH item.
- Countries of Origin: The country or countries where this ICH is recognized.
- Regions of Origin: The country or countries where this ICH is recognized.
- Title: The official title of the ICH item.
- Description: A detailed description of the ICH item, including relevant details.

# Your Task

Your task is it to generate high-quality question-answer pairs in a VQA style to assess the cultural knowledge of the intangible cultural heritage (ICH) item of state-of-the-art multimodal AI models. Be sure to follow the annotation guidelines provided below to ensure the quality and relevance of the question-answer pairs.

# Annotation Guidelines

## Question Requirements

Make sure the question meets all of the following requirements:

1. Clear and Concise
    The question is clear and concise and no longer than a single sentence.
2. Directly related to the ICH item
    The question is directly related to the ICH item.
3. Directly related to the visible content
    The question is directly related to the visible content in the image and requires visual analysis to answer.
4. Does not (partially) contain the answer
    The question does not contain any hints or clues to or parts of the answer that would make the answer obvious.
5. Does not contain subjective words
    The question does not contain subjective words like 'likely', 'possibly', 'probably', 'eventually', 'might', 'could', 'should', etc., which could introduce ambiguity.
6. Requires both image and cultural knowledge to answer
    The question requires both image and cultural knowledge to answer and is not answerable by looking only at the image or only knowing about the ICH item or reading the textual description.
7. (optional) Includes specific cultural terms
    The answer includes specific cultural terms, names, or phrases related to the ICH item. E.g., particular names mentioned in the description or parts of the title.

## Answer Requirements

Make sure the answer meets all of the following requirements:

1. Single Word or Multiword Expression
    The answer is a single word or multiword expression.
2. Clear, Objective, and Correct
    The answer is clear, objective, and unambiguously correct.
3. Directly Related to Visual Content
    The answer is directly related to the visual content of the image.
4. No General or Abstract Words
    The answer does not contain general, abstract, or non-depictable words like "Traditional", "Cooperation", "Gathering", "Solidarity”, “Community”, "Indoor", "Outdoor", "Urban", "Rural", etc.
5. Verifiable by Text and Image
    The answer is unambiguously verifiable by reading the textual information and inspecting the image.
6. (optional) Includes specific cultural terms
    The answer includes specific cultural terms, names, or phrases related to the ICH item. E.g., particular names mentioned in the description or parts of the title.

## Question Characteristics

### Target Aspects

Make sure the question targets different aspects of the ICH item, such as:

- Food
- Drinks
- Clothing
- Art
- Tools
- Sports
- Instruments
- Dance
- Music
- Rituals
- Traditions
- Festivals
- Customs
- Symbols
- Architecture
- Other

### Question Categories

Make sure the question falls into different categories, such as:

- Identification
    Questions that ask for the identification of objects, people, or elements in the image. E.g.: What is the name of the instrument shown in the image?
- Origin
    Questions that inquire about the origin or source of the CEF. E.g.: Which culture or country does this artifact belong to?
- Cultural Significance
    Questions that explore the cultural or religious significance of the depicted element. E.g.: What cultural or religious significance does this item hold in its native context?
- Function or Usage
    Questions that ask about the traditional or historical function or usage of the depicted element. E.g.: What was this object traditionally used for?
- Material and Craftsmanship
    Questions that focus on the materials used and the craftsmanship involved in creating the depicted element. E.g.: What material is used to construct this artifact?
- Location
    Questions that ask about the geographical location where the cultural event or facet takes place. E.g.: In which place does this dance take place?
- Symbolism
    Questions that delve into the symbolic meanings associated with the depicted element. E.g.: What does the color red symbolize in this cultural context?
- Historical
    Questions that relate to historical events or contexts depicted in the image. E.g.: What historical event is depicted in this image?
- Details
    Questions that ask for specific details about the formation, arrangement, or other aspects of the depicted element. E.g.: What formation are the dancers in?
- Other
    Questions that do not fall into the above categories but are relevant to the ICH item.

    
# Task Strategy

Before generating a question-answer pair, first think step-by-step and analyse the image:

1. What is visible in the image? Generate a highly detailed description of the key elements, objects, or people in the image. Take into account the textual description provided to identify details.
2. How does the visible content relate to the intangible cultural heritage item? Identify the connection between the contents of the image and the intangible cultural heritage item.

Then, think step-by-step about potential questions:

1. What can be asked about the image that is directly related to the visible content and the intangible cultural heritage item?
2. Can a concise and clear answer to the questions be inferred from the image and the provided information?

Finally, think step-by-step before generating the final question-answer pairs:

1. Does the question-answer pair strictly adhere to the guidelines provided above? Percisly check every part of the guidelines and drop the question-answer pair if it does not meet the criteria.
2. What aspect of the intangible cultural heritage item is targeted with the question?
3. What category does the question fall into?

# Output Format

For each question-answer pair, provide the following information in the following format:
```xml
<vqa-task>
    <image-analysis>
        <description>
            <!-- PUT YOUR DETAILED DESCRIPTION OF THE IMAGE HERE -->
        </description>
        <cultural-relatetness>
            <!-- PUT YOUR ANALYSIS OF HOW THE CONTENTS OF THE IMAGE RELATE TO THE INTANGIBLE CULTURAL HERITAGE ITEM HERE -->
        </cultural-relatetness>
    </image-analysis>
    <potential-questions>
        <qa-candidate>
            <question>
                <!-- PUT YOUR QUESTION HERE -->
            </question>
            <answer>
                <!-- PUT YOUR ANSWER HERE -->
            </answer>
            <guideline-adherence>
                <question-requirments>
                    <clear-and-concise>
                        <!-- YES OR NO -->
                    </clear-and-concise>
                    <directly-related-to-ich>
                        <!-- YES OR NO -->
                    </directly-related-to-ich>
                    <directly-related-to-visual-content>
                        <!-- YES OR NO -->
                    </directly-related-to-visual-content>
                    <does-not-contain-answer>
                        <!-- YES OR NO -->
                    </does-not-contain-answer>
                    <does-not-contain-subjective-words>
                        <!-- YES OR NO -->
                    </does-not-contain-subjective-words>
                    <requires-both-image-and-cultural-knowledge>
                        <!-- YES OR NO -->
                    </requires-both-image-and-cultural-knowledge>
                    <includes-specific-cultural-terms>
                        <!-- YES OR NO -->
                    </includes-specific-cultural-terms>
                </question-requirments>
                <answer-requirments>
                    <single-word-or-multiword-expression>
                        <!-- YES OR NO -->
                    </single-word-or-multiword-expression>
                    <clear-objective-and-correct>
                        <!-- YES OR NO -->
                    </clear-objective-and-correct>
                    <directly-related-to-visual-content>
                        <!-- YES OR NO -->
                    </directly-related-to-visual-content>
                    <no-general-or-abstract-words>
                        <!-- YES OR NO -->
                    </no-general-or-abstract-words>
                    <verifiable-by-text-and-image>
                        <!-- YES OR NO -->
                    </verifiable-by-text-and-image>
                    <includes-specific-cultural-terms>
                        <!-- YES OR NO -->
                    </includes-specific-cultural-terms>
                </answer-requirments>
            </guideline-adherence>
        </qa-candidate>
        ...
    </potential-questions>
    <final-qa-pairs>
        <!-- PUT ALL QA PAIRS THAT MEET ALL MANADATORY REQUIREMENTS HERE -->
        <qa-pair>
            <meets-requirements>
                <!-- DOES YOUR QUESTION-ANSWER PAIR MEET ALL MANDATORY REQUIREMENTS? YES OR NO -->
            </meets-requirements>
            <final-result-json>
                <!-- PUT YOUR FINAL RESULT AS JSON HERE -->
                {
                    "question": <insert question here>,
                    "answer": <insert answer here>,
                    "target_aspect": <insert target aspect here>
                    "question_category": <insert question category here>
                }
            </final-result-json>
        </qa-pair>
        ...
    </final-qa-pairs>
</vqa-task>
```

"""
