import re
import nltk
import spacy
from typing import List, Dict, Any
from dataclasses import dataclass, field
from nltk.tokenize import sent_tokenize
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

@dataclass
class TaskInfo:
    text: str
    entity: str = None
    deadline: str = None
    priority: str = 'normal'
    category: str = 'Uncategorized'
    context_score: float = field(default=0.0, compare=False)

class AdvancedTaskExtractor:
    def __init__(self):
        # Load NLP models and resources
        self.nlp = spacy.load("en_core_web_sm")
        nltk.download('punkt', quiet=True)
        
        # Comprehensive task and intent detection patterns
        self.task_patterns = {
            'commitment': [
                r'(?:has|have|need|needs|must|should|will|shall) to\b',
                r'\bmust\b',
                r'\bshould\b',
                r'\bneeds? to\b',
                r'\bis required to\b',
                r'\bis responsible for\b'
            ],
            'action_verbs': [
                r'\b(prepare|write|create|develop|submit|send|complete|finish|deliver)\b'
            ]
        }
        
        # Enhanced deadline recognition patterns
        self.deadline_patterns = [
            # Specific date references (like today, tomorrow, next week)
            r'\b(by|before|until|no later than) (?:today|tomorrow|next week|next month|next (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))',
            
            # Time-based deadlines (like "by 5 PM", "before noon")
            r'\b(by|before|until) (?:(?:\d{1,2}(?::\d{2})? ?(?:am|pm))|(?:noon|midnight))',
            
            # Duration-based deadlines (like "within 2 days", "in 3 hours")
            r'\b(within|in) \d+ (?:hour|day|week|month)s?'
        ]
        
        # Enhanced priority indicators
        self.priority_indicators = {
            'high': ['urgent', 'critical', 'asap', 'immediately', 'top priority'],
            'medium': ['soon', 'important', 'relatively urgent'],
            'low': ['when possible', 'if time permits', 'eventually']
        }

    def preprocess_text(self, text: str) -> List[str]:
        """Advanced text preprocessing with context preservation."""
        sentences = sent_tokenize(text)
        clean_sentences = [' '.join(sent.split()) for sent in sentences]
        return clean_sentences

    def is_potential_task(self, sentence: str) -> bool:
        """Comprehensive task identification."""
        lower_sentence = sentence.lower()
        
        # Check task indicator patterns
        task_indicators = self.task_patterns['commitment'] + self.task_patterns['action_verbs']
        return any(re.search(pattern, lower_sentence) for pattern in task_indicators)

    def extract_entities(self, text: str):
        """Extract named entities, ensuring 'Rahul' and similar names are classified as PERSON."""
        doc = self.nlp(text)
        entities = {"PERSON": [], "ORG": [], "GPE": [], "DATE": []}

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entities["PERSON"].append(ent.text)
            elif ent.label_ == "ORG":
                entities["ORG"].append(ent.text)
            elif ent.label_ == "GPE":
                entities["GPE"].append(ent.text)
            elif ent.label_ == "DATE":
                entities["DATE"].append(ent.text)

        return entities
    
    def replace_pronouns(self, text: str):
        """Replace pronouns with the last detected PERSON entity."""
        lines = text.split(". ")  # Splitting by sentence boundary
        last_person = None
        updated_lines = []

        for line in lines:
            doc = self.nlp(line)
            words = line.split()
            new_words = []

            for i, word in enumerate(words):
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        last_person = ent.text  # Update last known person

                if word.lower() in ["he", "she"] and last_person:
                    new_words.append(last_person)
                else:
                    new_words.append(word)

            updated_lines.append(" ".join(new_words))

        return ". ".join(updated_lines)
    
    def extract_deadline(self, sentence: str) -> str:
        """Extracts deadlines using predefined patterns."""
        for pattern in self.deadline_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                return match.group(0)  # Extracts and returns the deadline phrase
        return "No deadline"  # Default if no deadline is found

    def assign_task_to_entity(self, task: str, entities: dict):
        """Assign task to an entity based on context."""
        for entity in entities["PERSON"]:
            if entity.lower() in task.lower():
                return entity
        return entities["PERSON"][0] if entities["PERSON"] else "Unknown"
    
    def extract_and_assign_tasks(self, text: str):
        """Extract tasks and assign them to relevant entities with deadlines."""
        sentences = text.split(". ")
        tasks = []
        entities = self.extract_entities(text)

        for sentence in sentences:
            if re.search(r"\b(has to|must|should|need to|required to)\b", sentence, re.IGNORECASE):
                assigned_entity = self.assign_task_to_entity(sentence, entities)
                deadline = self.extract_deadline(sentence)

                tasks.append({
                    "task": sentence.strip(),
                    "assigned_to": assigned_entity,
                    "deadline": deadline,
                    "priority": "normal"
                })

        return tasks

    def score_entity_relevance(self, entities: Dict[str, List[str]], sentences: List[str]) -> Dict[str, float]:
        """Score entities based on their frequency and contextual relevance."""
        entity_scores = {}
        
        # Flatten entity list
        all_entities = [e for entity_list in entities.values() for e in entity_list]
        
        # Score entities based on:
        # 1. Frequency in text
        # 2. Proximity to task-related sentences
        # 3. Co-occurrence with task indicators
        for entity in all_entities:
            frequency = sum(entity.lower() in sent.lower() for sent in sentences)
            task_proximity = sum(self.is_potential_task(sent) and entity.lower() in sent.lower() 
                                  for sent in sentences)
            
            # Weighted scoring mechanism
            score = (frequency * 0.5) + (task_proximity * 1.5)
            entity_scores[entity] = score
        
        return entity_scores

    def select_primary_entity(self, entity_scores: Dict[str, float]) -> str:
        """Select the most relevant entity based on scoring."""
        if not entity_scores:
            return None
        
        # Sort entities by score in descending order
        sorted_entities = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_entities[0][0]

    def determine_priority(self, sentence: str) -> str:
        """Intelligent priority determination."""
        lower_sentence = sentence.lower()
        
        for priority, indicators in self.priority_indicators.items():
            if any(indicator in lower_sentence for indicator in indicators):
                return priority
        
        return 'normal'

    def semantic_categorization(self, tasks: List[TaskInfo]) -> Dict[str, List[TaskInfo]]:
        """Advanced semantic task categorization using TF-IDF and cosine similarity."""
        if not tasks:
            return {}
        
        task_texts = [task.text for task in tasks]
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(task_texts)
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Create semantic categories
        categories = defaultdict(list)
        category_prototypes = []
        
        for i, task in enumerate(tasks):
            # Find most similar existing category
            best_category_index = -1
            best_similarity = 0
            
            for j, prototype in enumerate(category_prototypes):
                similarity = similarity_matrix[i][prototype]
                if similarity > best_similarity and similarity > 0.3:
                    best_category_index = j
                    best_similarity = similarity
            
            # Create new category or assign to existing
            if best_category_index == -1:
                category_name = f"Category_{len(category_prototypes) + 1}"
                categories[category_name].append(task)
                category_prototypes.append(i)
            else:
                category_name = f"Category_{best_category_index + 1}"
                categories[category_name].append(task)
        
        return dict(categories)

    def process_text(self, text: str) -> Dict[str, Any]:
        """Comprehensive text processing with automatic entity extraction."""
        sentences = self.preprocess_text(text)
        
        # Extract all possible entities
        extracted_entities = self.extract_entities(text)
        
        # Score entity relevance
        entity_scores = self.score_entity_relevance(extracted_entities, sentences)
        
        # Select primary entity
        primary_entity = self.select_primary_entity(entity_scores)
        
        tasks = []
        for sentence in sentences:
            if self.is_potential_task(sentence):
                task = TaskInfo(
                    text=sentence,
                    entity=primary_entity,  # Dynamically assign most relevant entity
                    deadline=self.extract_deadline(sentence),
                    priority=self.determine_priority(sentence)
                )
                tasks.append(task)
        
        categorized_tasks = self.semantic_categorization(tasks)
        
        return {
            'tasks': tasks,
            'categorized_tasks': categorized_tasks,
            'extracted_entities': extracted_entities,
            'primary_entity': primary_entity
        }

# Example usage
def main():
    sample_text = """ Rahul wakes up early every day. He goes to college in the morning and comes back at 3 pm. At present, Rahul is outside. He has to buy the snacks for all of us.  Rahul should clean the room by today. He should go home by 5 pm."""
    
    extractor = AdvancedTaskExtractor()
    results = extractor.process_text(sample_text)
    
    print("\nExtracted Entities:", results['extracted_entities'])
    print("\nPrimary Entity:", results['primary_entity'])
    
    print("\nExtracted Tasks:")
    for task in results['tasks']:
        print(f"\nTask: {task.text}")
        print(f"Assigned to: {task.entity or 'Unassigned'}")
        print(f"Deadline: {task.deadline or 'No deadline'}")
        print(f"Priority: {task.priority}")
    
    print("\nCategorized Tasks:")
    for category, tasks in results['categorized_tasks'].items():
        print(f"\n{category}:")
        for task in tasks:
            print(f"- {task.text}")

if __name__ == "__main__":
    main()
