"""Distinctive words for the map: what each item is about, and what links two items.

Each main node (a source, or a memory that isn't a source's own summary) gets its most
distinctive words, scored by TF-IDF over everything saved: frequent in that item, rare
elsewhere. Two items whose distinctive words overlap get a "bridge" naming the words
they share. Common English, conversational filler (podcast transcripts are full of
"yeah" and "going"), and programming keywords are ignored so the words reflect topics.
"""
from __future__ import annotations

import math
import re
from collections import Counter

WORD = re.compile(r"[A-Za-z][A-Za-z'\-]{2,}")

STOP = set('''
a about above after again against all almost also although always am among an and another any anybody anyone
anything anyway anywhere are aren't around as at away back be became because become becomes been before being
below beside besides best better between beyond both but by came can can't cannot come comes could couldn't did
didn't do does doesn't doing don't done down during each either else enough even ever every everybody everyone
everything everywhere few for from further get gets getting give given gives go goes going gone got gotten had
hadn't has hasn't have haven't having he he's her here here's hers herself him himself his how however i i'd
i'll i'm i've if in inside instead into is isn't it it's its itself just keep kept kind know known knows last
least less let let's like likely made make makes making many may maybe me mean means might mine more most mostly
much must my myself near need needs neither never next no nobody none nor not nothing now nowhere of off often
oh ok okay on once one ones only onto or other others otherwise our ours ourselves out over own part per perhaps
put quite rather really right said same say saying says see seem seemed seems seen several shall she she's
should shouldn't since so some somebody someone something sometimes somewhere soon still such sure take taken
takes taking than that that's the their theirs them themselves then there there's these they they'd they'll
they're they've thing things think this those though through thus till to together too took toward towards
under unless until up upon us use used uses using very via want wanted wants was wasn't way ways we we'd we'll
we're we've well went were weren't what what's whatever when whenever where wherever whether which while who
who's whoever whole whom whose why will with within without won't would wouldn't yes yet you you'd you'll you're
you've your yours yourself yourselves
actually alright anyway basically big bit definitely exactly gonna gotta guess guys hey huh kinda lot lots
literally mhm okay pretty probably sort stuff talk talking tell telling thank thanks totally uh um wanna
whatnot yeah yep yup look looking looked feel felt good great little long new old real thought thinking
able across already among around ask asked bring brought call called day days first found getting home
hour hours later life live man men minute minutes month months people person place point points problem
question questions second seconds start started thing times today told try trying turn turned two three four
five six seven eight nine ten hundred thousand million billion week weeks woman women work worked working
world year years
def self return import from class const var let function async await true false null none undefined elif
else except finally raise try catch throw new this typeof instanceof void public private protected static
string int str dict list tuple bool float len args kwargs param params value values key keys item items
data type types object objects array arrays true false http https www com org net html css div span src href
px rem em todo fixme
file files lines line row rows passage passages source sources dataset datasets split splits config default
train test validation part parts image images audio video frame frames transcript captions caption
'''.split())


STOP |= set('''
laughter applause music cuz etc interesting weird remember various plausible whatever
anyways somewhat literally obviously seriously honestly crazy amazing awesome cool nice fine kinda
text description sentence example examples note notes api url urls json version update updated error errors
post get set add remove create created default option options section content contents page pages
'''.split())


def tokens(text):
    """(word, named) pairs; `named` is True when the word is written like a name or acronym:
    capitalized mid-sentence, or in all capitals."""
    for match in WORD.finditer(text):
        raw = match.group()
        word = raw.lower().strip("'-")
        if word.endswith("'s"):
            word = word[:-2]
        if len(word) < 3 or word in STOP or word.isdigit():
            continue
        before = text[max(0, match.start() - 12):match.start()].rstrip()[-1:]  # a short look back
        mid_sentence = before not in ('', '.', '!', '?', '\n', '"', ':', '#', '-', '*', '>', '|', '[', '(')
        named = (len(raw) > 1 and raw.isupper()) or (raw[0].isupper() and mid_sentence)
        yield word, named


class Stats:
    """Word counts per item and how many items use each word."""

    def __init__(self, documents, max_chars=400_000):
        self.counts, named = {}, Counter()
        for key, text in documents.items():
            counter = Counter()
            for word, is_named in tokens(text[:max_chars]):
                counter[word] += 1
                named[word] += is_named
            self.counts[key] = counter
        self.frequency = Counter()
        for counter in self.counts.values():
            self.frequency.update(counter.keys())
        uses = Counter()
        for counter in self.counts.values():
            uses.update(counter)
        # Names, programs, and acronyms are what tie topics together: up to 3x weight.
        self.boost = {word: 1 + 2 * named[word] / uses[word] for word in uses}
        self.total = len(self.counts)

    def idf(self, word):
        return math.log(1 + self.total / self.frequency[word])


def profiles(documents, top=8):
    """documents: {key: text}. Returns ({key: [(word, weight 0..1)]} with the `top` most
    distinctive words of each, and the Stats used, which bridges() also needs)."""
    stats = Stats(documents)
    result = {}
    for key, counter in stats.counts.items():
        scored = {word: (1 + math.log(n)) * stats.idf(word) * stats.boost[word]
                  for word, n in counter.items() if n >= 2 or stats.total <= 2 or len(counter) < 200}
        best = distinct(sorted(scored.items(), key=lambda item: -item[1]), top)
        peak = best[0][1] if best else 1
        result[key] = [(word, round(score / peak, 3)) for word, score in best]
    return result, stats


def distinct(ranked, count):
    """The first `count` (word, score) pairs, skipping simple plural/singular repeats."""
    chosen, stems = [], set()
    for word, score in ranked:
        stem = word[:-1] if word.endswith('s') and len(word) > 3 else word
        if stem in stems:
            continue
        stems.add(stem)
        chosen.append((word, score))
        if len(chosen) == count:
            break
    return chosen


def bridges(stats, limit=16, per_node=4, words_shown=3, skip=frozenset(), vocabulary=400, min_ratio=0.35):
    """Pairs of items that talk about the same things: [{a, b, words, shared, score}],
    strongest first. A shared word counts by how much both items use it, discounted when
    most items use it. `skip` holds pairs (a, b) that shouldn't be bridged, e.g. a source
    and its own summary memory."""
    common = {key: {word: n for word, n in counter.most_common(vocabulary)}
              for key, counter in stats.counts.items()}
    crowded = 0.6 * stats.total if stats.total >= 5 else stats.total + 1
    keys = list(common)
    candidates = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if (a, b) in skip or (b, a) in skip:
                continue
            words_a, words_b = common[a], common[b]
            # Long texts must use a word more than once for it to count as a shared topic.
            need_a = 2 if sum(words_a.values()) > 300 else 1
            need_b = 2 if sum(words_b.values()) > 300 else 1
            shared = [(word, min(math.log(1 + words_a[word]), math.log(1 + words_b[word]))
                       * stats.idf(word) * stats.boost[word])
                      for word in words_a.keys() & words_b.keys()
                      if words_a[word] >= need_a and words_b[word] >= need_b and stats.frequency[word] < crowded]
            if len(shared) < 2:
                continue
            shared = distinct(sorted(shared, key=lambda item: -item[1]), len(shared))
            score = sum(weight for _, weight in shared[:5])
            candidates.append({'a': a, 'b': b, 'words': [w for w, _ in shared[:words_shown]],
                               'shared': len(shared), 'score': round(score, 3)})
    candidates.sort(key=lambda bridge: -bridge['score'])
    # Only bridges close to the strongest: weak overlaps are mostly everyday words.
    floor = candidates[0]['score'] * min_ratio if candidates else 0
    candidates = [bridge for bridge in candidates if bridge['score'] >= floor]
    chosen, degree = [], Counter()
    for bridge in candidates:
        if degree[bridge['a']] >= per_node or degree[bridge['b']] >= per_node:
            continue
        chosen.append(bridge)
        degree[bridge['a']] += 1
        degree[bridge['b']] += 1
        if len(chosen) == limit:
            break
    return chosen
