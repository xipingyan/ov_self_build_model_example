
#pragma once
#include <string>
#include <vector>

class Profile
{
public:
	Profile() : _active(false) {}
	Profile(const std::string &name,
			const char* file, int line,
			const std::vector<std::pair<std::string, std::string>> &args =
				std::vector<std::pair<std::string, std::string>>());
	~Profile();

	static bool enabled();

private:
	std::string _name;
	const char *_file = nullptr;
	int _line = 0;
	uint64_t _ts1;
	std::vector<std::pair<std::string, std::string>> _args;
	bool _active = true;
};

/* Usage example:
 * ==========================================
 * Profile(var, "fun_name")
 *
 * Usage 2: specific scope
 * {
 *    Profile(p, "fun_name")
 *    func()
 * }
 *
 * Usage 3: specific scope with some params.
 * {
 *    Profile(p2, "fun_name", {{"arg1", "sleep 30 ms"}});
 *    func()
 * }
 */
#define PROFILE(VAR, NAME) auto VAR = Profile::enabled() ? Profile(NAME, __FILE__, __LINE__) : Profile()
#define PROFILE_ARGS(VAR, NAME, ...) auto VAR = Profile::enabled() ? Profile(NAME, __FILE__, __LINE__, __VA_ARGS__) : Profile()