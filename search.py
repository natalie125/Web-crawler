import requests
from bs4 import BeautifulSoup
import time
import json
from urllib.parse import urlparse, urljoin
from argparse import ArgumentParser
import re

inverted_index = {}
page_graph = {}
index_file_name = "index.json"

# Check if the URL belongs to the same domain
def check_valid_url(url, start_domain):
    parsed_url = urlparse(url)
    parsed_start_url = urlparse(start_domain)
    return (parsed_url.scheme in ["http", "https"]) and (parsed_start_url.netloc == parsed_url.netloc)

# Clean and tokenize text
def clean_text(text):
    return re.findall(r'\b[a-z]{2,}\b', text.lower())

# Crawl the website and build the index
def crawl_and_build_index(start_url, max_pages, delay):
    visited_urls = set()
    urls_to_visit = [start_url]
    pages_crawled = 0

    while urls_to_visit and pages_crawled < max_pages:
        current_url = urls_to_visit.pop(0)
        if current_url in visited_urls:
            continue

        print(f"Preparing to crawl: {current_url}")  # Log before making the request
        try:
            response = requests.get(current_url, timeout=10)
            print(f"Received response from {current_url} with status: {response.status_code}")  # Log response status
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            text = soup.get_text()
            words = clean_text(text)
            for position, word in enumerate(words):
                if word not in inverted_index:
                    inverted_index[word] = {current_url: [position]}
                else:
                    if current_url not in inverted_index[word]:
                        inverted_index[word][current_url] = [position]
                    else:
                        inverted_index[word][current_url].append(position)

            # Processing links on the page
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(current_url, href)
                if check_valid_url(full_url, start_url) and full_url not in visited_urls:
                    urls_to_visit.append(full_url)
                    add_link_to_graph(current_url, full_url)

            visited_urls.add(current_url)
            pages_crawled += 1
            print(f"Crawled {current_url}, total pages crawled: {pages_crawled}")  # Log after successful crawl
            time.sleep(delay)
        except requests.RequestException as e:
            print(f"Failed to process {current_url}: {str(e)}")  # Log request errors

        except Exception as e:
            print(f"An unexpected error occurred while processing {current_url}: {str(e)}")

# Save the index
def save_index_to_file():
    data = {
        'inverted_index': inverted_index,
        'page_graph': {source: list(links) for source, links in page_graph.items()}
    }
    with open(index_file_name, 'w') as file:
        json.dump(data, file, indent=4)

# Load the index
def load_index_from_file():
    global inverted_index, page_graph
    try:
        with open(index_file_name) as file:
            data = json.load(file)
        inverted_index = data.get('inverted_index', {})
        page_graph = {source: set(links) for source, links in data.get('page_graph', {}).items()}
    except FileNotFoundError:
        print("Index file not found. Please build the index first.")

# Adding a link to the page graph
def add_link_to_graph(source_url, target_url):
    if source_url not in page_graph:
        page_graph[source_url] = set()
    page_graph[source_url].add(target_url)

# Cal the Page rank for each web page
def calculate_pagerank(iterations=100, d=0.85):
    pageranks = {page: 1.0 / len(page_graph) for page in page_graph}
    for _ in range(iterations):
        new_ranks = {}
        for page in page_graph:
            new_rank = (1 - d) / len(page_graph)
            for referring_page in page_graph:
                if page in page_graph[referring_page]:
                    new_rank += d * pageranks[referring_page] / len(page_graph[referring_page])
            new_ranks[page] = new_rank
        pageranks = new_ranks
    return pageranks

# Find pages containing the words
def find_pages(search_terms):
    page_ranks = calculate_pagerank()  # Calculate PageRank for all pages
    terms = search_terms.lower().split()
    document_scores = {}
    proximity_boost = 10

    # Check if multiple terms and initialize document tracking for each term
    term_positions = {term: {} for term in terms}

    # Collect positions for each term
    for term in terms:
        if term in inverted_index:
            for doc, positions in inverted_index[term].items():
                term_positions[term][doc] = positions

    # Calculate initial score based on word frequency, positions, and proximity
    for doc in set().union(*[data.keys() for data in term_positions.values()]):
        document_scores[doc] = 0
        previous_term_positions = None

        for term in terms:
            if doc in term_positions[term]:
                current_positions = term_positions[term][doc]
                term_frequency = len(current_positions)
                normalized_position_score = sum(1 / (pos + 1) for pos in current_positions)  # Higher score for terms closer to beginning
                document_scores[doc] += term_frequency * normalized_position_score

                # Check proximity if the it is not none
                if previous_term_positions is not None:
                    # Calculate proximity bonus
                    proximity_bonus = sum(1 for prev_pos in previous_term_positions for curr_pos in current_positions if abs(prev_pos - curr_pos) == 1)
                    document_scores[doc] += proximity_boost * proximity_bonus

                previous_term_positions = current_positions

    # Combine all the score
    for doc in document_scores:
        document_scores[doc] += page_ranks.get(doc, 0) * 0.5  # The weight of PageRank influence

    # Sort documents by their final score
    sorted_documents = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_documents

# Print index entries
def print_index(word):
    word = word.lower()
    if word in inverted_index:
        print(f"Inverted index for '{word}':")
        for url, positions in inverted_index[word].items():
            print(f"  {url}: Positions {positions}")
    else:
        print(f"No entry found for '{word}'.")

def setup_argparser():
    """ Setup the command line argument parser """
    parser = ArgumentParser(description="Search Tool CLI")
    parser.add_argument("command", choices=['build', 'load', 'print', 'find'], help="Command to run")
    parser.add_argument("search_terms", nargs="?", help="Search terms for 'print' and 'find' commands", type=str, default="")
    parser.add_argument("--max-pages", help= "Maximum number of pages to crawl", type=int, default=float('inf'))   #Crawl all the pages
    parser.add_argument("--delay", help="Delay between requests in seconds", type=int, default=6)
    return parser

def main():
    """ Main function to run the search tool """
    parser = setup_argparser()
    args = parser.parse_args()

    if args.command == "build":
        start_url = "https://quotes.toscrape.com/"
        crawl_and_build_index(start_url, args.max_pages, args.delay)
        save_index_to_file()
    elif args.command == "load":
        load_index_from_file()
    elif args.command == "print":
        load_index_from_file()
        print_index(args.search_terms)
    elif args.command == "find":
        load_index_from_file()
        sorted_documents = find_pages(args.search_terms)
        print(f"Pages containing '{args.search_terms}':")
        for doc_id, score in sorted_documents:
            print(f"  {doc_id}: Score = {score}")

if __name__ == "__main__":
    main()
