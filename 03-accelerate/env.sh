# 可以通过 module load 的方式加载你需要的对象，如：module load xxxx
# 计算节点的 /usr/local 没有 NumPy。评测容器能直接 import 时不要改搜索路径。
# prepare 会把 PYTHONPATH 设成 src，计时 worker 会换成临时 HOME。
# .pth 保住前者，export 保住后者。
_numpy_site="${HOME}/.local/hellohpc-py/lib/python3.9/site-packages"
if [ -d "${_numpy_site}/numpy" ]; then
  if ! PYTHONPATH= PYTHONNOUSERSITE=1 python3 -c 'import numpy' >/dev/null 2>&1; then
    case ":${PYTHONPATH:-}:" in
      *":${_numpy_site}:"*) ;;
      *) export PYTHONPATH="${_numpy_site}${PYTHONPATH:+:${PYTHONPATH}}" ;;
    esac
    _user_site="$(python3 -c 'import site; print(site.getusersitepackages())')"
    mkdir -p "${_user_site}"
    printf '%s\n' "${_numpy_site}" > "${_user_site}/hellohpc-numpy.pth"
    unset _user_site
  fi
fi
unset _numpy_site
