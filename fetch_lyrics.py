import lyricsgenius
import re

GENIUS_TOKEN = "BoRySZQNsJPQoB_aMkbPZth5EDga2xmvsFj1YJPnVtVRPqzrD91JUy6ttl-Ld_nn"
ARTIST_NAME = "LINKIN PARK"  
MAX_SONGS = 100

def clean_lyrics(text):
    text = re.sub(r'\[.*?\]', '', text)   
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.lower().strip()
    return text

def fetch_and_save():
    genius = lyricsgenius.Genius(
        GENIUS_TOKEN,
        skip_non_songs=True,
        excluded_terms=['(Remix)', '(Live)'],
        timeout=20,
        retries=3,
        sleep_time=1
    )
    print(f'Fetching lyrics for {ARTIST_NAME}...')
    artist = genius.search_artist(
        ARTIST_NAME,
        max_songs=MAX_SONGS,
        get_full_info=False
    )

    print(f'Found {len(artist.songs)} songs')

    all_lyrics = []

    for i, song in enumerate(artist.songs, start=1):
        print(f'[{i}/{len(artist.songs)}] Processing: {song.title}')

        if song.lyrics:
            cleaned = clean_lyrics(song.lyrics)
            all_lyrics.append(cleaned)

    combined = "\n\n".join(all_lyrics)
    with open("lyrics.txt", "w", encoding="utf-8") as f:
        f.write(combined)

    print(f'Saved {len(all_lyrics)} songs to lyrics.txt')
    print(f'Total characters: {len(combined):,}')

if __name__ == '__main__':
    fetch_and_save()