#!/usr/bin/env python3
"""
从 environment-linux.yml 生成 requirements.txt
YAML 中的版本号优先级最高
未指定版本号时查询当前环境的实际版本
排除 python 和 pip
"""

import os
from datetime import datetime
import yaml
from importlib.metadata import version, PackageNotFoundError


# 排除的包（不加入 requirements.txt）
EXCLUDE_PACKAGES = {"python", "pip"}


def get_installed_version(package_name):
    """获取包的安装版本，未安装返回 None"""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def parse_package_string(dep):
    """
    解析包字符串，返回 (包名, yaml版本号或None)
    例如: "tensorflow=2.15.0" -> ("tensorflow", "2.15.0")
          "numpy" -> ("numpy", None)
    """
    if "=" in dep:
        parts = dep.split("=")
        pkg_name = parts[0]
        pkg_version = parts[1]
        return pkg_name, pkg_version
    else:
        return dep, None


def parse_environment_yml(filepath):
    """解析 environment-linux.yml，提取包列表和版本信息"""
    with open(filepath, "r") as f:
        env = yaml.safe_load(f)

    packages = []

    for dep in env.get("dependencies", []):
        if isinstance(dep, str):
            # 简单字符串格式：package 或 package=version
            pkg_name, yaml_version = parse_package_string(dep)
            if pkg_name not in EXCLUDE_PACKAGES:
                packages.append((pkg_name, yaml_version))
        elif isinstance(dep, dict) and "pip" in dep:
            # pip 子列表
            for pip_dep in dep["pip"]:
                pkg_name, yaml_version = parse_package_string(pip_dep)
                if pkg_name not in EXCLUDE_PACKAGES:
                    packages.append((pkg_name, yaml_version))

    return packages


def main():
    yml_file = "environment-linux.yml"
    output_file = "requirements.txt"

    print(f"读取 {yml_file}...")
    packages = parse_environment_yml(yml_file)
    print(f"发现 {len(packages)} 个包（排除 {EXCLUDE_PACKAGES}）")

    lines = []
    for pkg_name, yaml_version in packages:
        if yaml_version:
            # YAML 中有版本号，优先使用
            lines.append(f"{pkg_name}=={yaml_version}")
            print(f"  ✓ {pkg_name}=={yaml_version} (来自 YAML)")
        else:
            # YAML 中没有版本号，查询当前环境
            env_version = get_installed_version(pkg_name)
            if env_version:
                lines.append(f"{pkg_name}=={env_version}")
                print(f"  ✓ {pkg_name}=={env_version} (来自当前环境)")
            else:
                lines.append(pkg_name)
                print(f"  ⚠ {pkg_name} (未安装，无版本号)")

    # 添加头部注释
    header_lines = [
        f"# Generated from {yml_file}",
        f"# Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# Environment: {os.environ.get('ENV', 'unknown')}",
        "#",
    ]

    # 合并所有行
    all_lines = header_lines + lines

    with open(output_file, "w") as f:
        f.write("\n".join(all_lines) + "\n")

    print(f"\n已生成 {output_file}：")
    print("-" * 40)
    print("\n".join(all_lines))
    print("-" * 40)


if __name__ == "__main__":
    main()
