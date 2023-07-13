class Media {
	constructor(url) {
		this._url = url;
	}

	play() {
		console.log(this._url);
	}
}

class Movie extends Media {
	constructor(url, title, releasedDate, duration) {
		super(url);
		this._title = title;
		this._releasedDate = releasedDate;
		history._duration = duration;
	}
}

class TvShow extends Media {
	constructor(url, title, numberOfEpisodePerSeason, numberOfSeasons) {
		super(url);
		this._title = title;
		this._numberOfEpisodePerSeason = numberOfEpisodePerSeason;
		this._numberOfSeasons = numberOfSeasons;
	}
}

const PredatorMovie = new Movie(
	'https://fr.wikipedia.org/wiki/Predator_(film)',
	'Predator',
	1987,
	107
);

const TerminatorMovie = new Movie('Terminator', 1984, 107);
const AlienMovie = new Movie('Alien', 1979, 117);

const FriendsTvShow = new TvShow('Friends', 23, 10);
const ScrubsTvShow = new TvShow('Scrubs', 20, 9);
const CougarTownTvShow = new TvShow('Cougar Town', 13, 5);

PredatorMovie.play();
