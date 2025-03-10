"""
Ramel Mirza, 000778681

This script loads in the coolPHDpapers.csv file which contains CS papers
based off of AI/ML and manages dialog with the user using a recommender system

"""
from csv import DictReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_articles(file_name):
    """
    Loads the articles from the coolPHDpapers.csv file into a list (You can see the notes all the way at the bottom of this script)
    :param file_name: Name of the file
    :return: List of the articles containing its terms, title, and abstract intro/details
    """
    articles = []
    with open(file_name, encoding="utf-8") as csv_file:
        seen = set()
        reader = DictReader(csv_file)
        for row in reader:
            uniques = (row["titles"], row["abstracts"])
            if uniques not in seen:
                seen.add(uniques)
                articles.append(row)
    return articles


def show_article(articles, user_input):
    """
    Displays the article that the user input
    :param articles: All the articles
    :param user_input: What article # the user chose, but -1 in the actual index because articles start from 1
    :return: None
    """
    title_of_article = articles[user_input - 1]["titles"]
    print("\n")
    print(f"\033[94m{user_input}) {title_of_article}\033[0m")
    abstract_of_article = articles[user_input - 1]["abstracts"]
    print("\n")
    print(abstract_of_article)


def get_vectors(articles):
    """
    Turns the body or abstract introduction to the paper into its vector representation
    :param articles: All the articles
    :return: All the vectors
    """
    abstracts_only = []
    for a in articles:
        abstracts_only.append(a["abstracts"])
    # I originally didn't use ngrams, but then I realized "neural network" or "deep learning" or "reinforcement learning" should be counted as a whole.
    vectorizer = TfidfVectorizer(min_df=0.03, max_df=0.33, stop_words="english", ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(abstracts_only)
    return vectors


def similarities_most(chosen_article, vectors):
    """
    Finds the 8 most similar articles
    :param chosen_article: The article that the user chose
    :param vectors: Vectors of each article (used in cosine_similarity function)
    :return: List containing the indices of the 3 least similar articles
    """
    cos_sims = cosine_similarity(chosen_article, vectors)
    all_sims = cos_sims[0]
    most_similar_articles = []
    for i in range(0, 8):
        most_similar_value = 0
        most_similar_index = 0
        for j in range(0, len(all_sims)):
            if j in most_similar_articles: # If the greatest value is already in the list, use the next smallest value instead
                continue
            if all_sims[j] >= 0.97:
                continue
            else:
                if all_sims[j] > most_similar_value:
                    most_similar_value = all_sims[j]
                    most_similar_index = j

        most_similar_articles.append(most_similar_index)

    return most_similar_articles


def similarities_least(chosen_article, vectors):
    """
    Finds the 3 least similar articles
    :param chosen_article: The article that the user chose
    :param vectors: Vectors of each article (used in cosine_similarity function)
    :return: List containing the indices of the 3 least similar articles
    """
    cos_sims = cosine_similarity(chosen_article, vectors)
    all_sims = cos_sims[0]
    least_similar_articles = []
    for i in range(0, 3):
        least_similar_value = 2.00
        least_similar_index = 0
        for j in range(0, len(all_sims)):
            if j in least_similar_articles: # If the least value is already in the list, use the next smallest value instead
                continue
            else:
                if all_sims[j] < least_similar_value:
                    least_similar_value = all_sims[j]
                    least_similar_index = j

        least_similar_articles.append(least_similar_index)

    return least_similar_articles


def show_similarities(articles, most_similar, least_similar):
    """
    Displays the 8 most similar articles, and the 3 least similar articles
    :param articles: The full dictionary of each article
    :param most_similar: Indices of the most similar articles based on what article the user entered
    :param least_similar: Indices of the least similar articles based on what article the user entered
    :return: None
    """
    print("\n")
    print("Here are some similar papers you should look at: \n")
    most_similar = sorted(most_similar)
    for ms in most_similar:
        print(f"\033[94m{ms + 1}) {articles[ms]["titles"]}\033[0m")

    print("\n")
    print("Here are some NOT so similar papers: \n")
    for ms in least_similar:
        print(f"\033[94m{ms + 1}) {articles[ms]["titles"]}\033[0m")


def main():
    """
    Uses a recommender system to print out similar/not-so-similar computer science papers and create dialog with the user
    :return: None
    """
    articles = load_articles("data/coolPHDpapers.csv")
    vectors = get_vectors(articles)
    print("\n")
    print("==============================================================")
    print(f"\033[94mLoaded {len(articles)} articles from cool computer scientists\033[0m")
    print("==============================================================")
    print("\n")
    print("The numbers before each title indicate the article ID.\nYou can type the id to get the abstract details of the article.\nYou can also enter '-1' to exit the recommender system whenever you want.")

    while True:
        try:
            print("\n")
            user_input = int(input("Pick an article:\n    > "))
            if user_input == -1:
                print("\n")
                print("Exiting the recommender system.")
                break
            elif user_input == 0:
                print("\n")
                print("Articles start from 1, not 0...try again")
            elif 0 < user_input <= len(articles):
                show_article(articles, user_input)
                most_similar_articles = similarities_most(vectors[user_input - 1], vectors)
                least_similar_articles = similarities_least(vectors[user_input - 1], vectors)
                show_similarities(articles, most_similar_articles, least_similar_articles)

        except ValueError:
            print("\n")
            print("Invalid input, try again...\n    > ")


if __name__ == "__main__":
    main()