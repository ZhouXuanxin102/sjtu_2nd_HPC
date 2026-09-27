#include <cstdlib>
#include <deque>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sched.h>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "main.hpp"

using namespace std;

unsigned load_block(istream& i, uint32 block[]);
void save_block(ostream& o, const uint32 block[]);
bool find_collision(const uint32 IV[], uint32 msg1block0[], uint32 msg1block1[], uint32 msg2block0[], uint32 msg2block1[], bool verbose);

namespace {

const uint32 kMD5IV[4] = { 0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476 };

struct Job {
	string in, o1, o2;
	uint32 IV[4];
	string prefix;
	atomic<int> workers;
	atomic<int> done;
	Job() : workers(0), done(0) {}
};

int claim_job(deque<Job>& jobs, mutex& mu)
{
	lock_guard<mutex> lock(mu);
	int best = -1;
	int best_h = 1000000;
	bool gang = getenv("MINICLASH_GANG") != 0;
	for (int i = 0; i < (int)jobs.size(); ++i) {
		if (jobs[i].done.load())
			continue;
		if (gang) {
			best = i;
			break;
		}
		int h = jobs[i].workers.load();
		if (h < best_h) {
			best = i;
			best_h = h;
		}
	}
	if (best >= 0)
		jobs[best].workers.fetch_add(1);
	return best;
}

bool prepare_job(Job& job)
{
	ifstream ifs(job.in.c_str(), ios::binary);
	if (!ifs)
		return false;
	for (int i = 0; i < 4; ++i)
		job.IV[i] = kMD5IV[i];
	uint32 block[16];
	job.prefix.clear();
	while (true) {
		unsigned len = load_block(ifs, block);
		if (!len)
			break;
		for (unsigned k = 0; k < 16; ++k)
			for (unsigned c = 0; c < 4; ++c)
				job.prefix.push_back((char)((block[k] >> (c * 8)) & 0xFF));
		md5_compress(job.IV, block);
	}
	return true;
}

bool write_pair(const Job& job, const uint32 b0[], const uint32 b1[], const uint32 c0[], const uint32 c1[])
{
	ofstream o1(job.o1.c_str(), ios::binary);
	ofstream o2(job.o2.c_str(), ios::binary);
	if (!o1 || !o2)
		return false;
	o1.write(job.prefix.data(), (streamsize)job.prefix.size());
	o2.write(job.prefix.data(), (streamsize)job.prefix.size());
	save_block(o1, b0);
	save_block(o1, b1);
	save_block(o2, c0);
	save_block(o2, c1);
	return o1 && o2;
}

} // namespace

int run_batch(const string& tasks_path, int threads)
{
	ifstream in(tasks_path.c_str());
	if (!in) {
		cerr << "cannot read task file: " << tasks_path << endl;
		return 1;
	}
	vector<string> lines;
	string line;
	while (getline(in, line)) {
		if (!line.empty())
			lines.push_back(line);
	}
	deque<Job> jobs;
	for (size_t n = 0; n < lines.size(); ++n) {
		istringstream iss(lines[n]);
		string f1, f2, f3, extra;
		if (!(iss >> f1 >> f2 >> f3) || (iss >> extra)) {
			cerr << "malformed line: " << lines[n] << endl;
			return 1;
		}
		jobs.emplace_back();
		jobs.back().in = f1;
		jobs.back().o1 = f2;
		jobs.back().o2 = f3;
	}
	for (size_t i = 0; i < jobs.size(); ++i) {
		if (!prepare_job(jobs[i])) {
			cerr << "cannot open inputfile: " << jobs[i].in << endl;
			return 1;
		}
	}
	if (jobs.empty()) {
		cout << "run: generated 0 collisions" << endl;
		return 0;
	}
	if (threads < 1)
		threads = 1;
	if (threads > 32)
		threads = 32;

	mutex mu;
	atomic<unsigned> seq(1);
	vector<int> cpus;
	cpu_set_t allowed;
	CPU_ZERO(&allowed);
	if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0) {
		for (int c = 0; c < CPU_SETSIZE; ++c)
			if (CPU_ISSET(c, &allowed))
				cpus.push_back(c);
	}
	if (!cpus.empty() && threads > (int)cpus.size())
		threads = (int)cpus.size();
	cerr << "threads=" << threads << " jobs=" << jobs.size() << " cpus=" << cpus.size() << endl;

	vector<thread> pool;
	pool.reserve(threads);
	for (int t = 0; t < threads; ++t) {
		pool.emplace_back([&, t]() {
			if (!cpus.empty()) {
				cpu_set_t cs;
				CPU_ZERO(&cs);
				CPU_SET(cpus[t % cpus.size()], &cs);
				sched_setaffinity(0, sizeof(cs), &cs);
			}
			while (true) {
				int i = claim_job(jobs, mu);
				if (i < 0)
					return;
				Job& job = jobs[i];
				g_stop = &job.done;
				while (!job.done.load()) {
					unsigned id = seq.fetch_add(1);
					seed32_1 = 10007u + id * 7919u;
					seed32_2 = 30011u + id * 104729u;
					uint32 a0[16], a1[16], b0[16], b1[16];
					if (find_collision(job.IV, a0, a1, b0, b1, false)) {
						if (!job.done.exchange(1))
							write_pair(job, a0, a1, b0, b1);
						break;
					}
				}
				g_stop = 0;
				job.workers.fetch_sub(1);
			}
		});
	}
	for (size_t t = 0; t < pool.size(); ++t)
		pool[t].join();

	int ok = 0;
	for (size_t i = 0; i < jobs.size(); ++i) {
		ifstream check(jobs[i].o1.c_str(), ios::binary);
		if (jobs[i].done.load() && check)
			++ok;
	}
	cout << "run: generated " << ok << " collisions" << endl;
	return ok == (int)jobs.size() ? 0 : 1;
}
