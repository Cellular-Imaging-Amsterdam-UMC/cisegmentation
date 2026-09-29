from cisegmentation import resources


def test_quantile_operation_limit_is_recoverable_without_being_a_memory_oom():
    error = RuntimeError("quantile() input tensor is too large")
    assert resources.tile_size_error(error)
    assert not resources.memory_error(error)
    assert not resources.tile_size_error(RuntimeError("tensor shape does not match"))


def test_slurm_memory_allocation_bounds_host_ram(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "16384")
    monkeypatch.delenv("SLURM_MEM_PER_CPU", raising=False)
    monkeypatch.setattr(resources, "cgroup_directories", list)
    monkeypatch.setattr(resources, "_job_rss", lambda: resources.GIB)
    result = resources.snapshot()
    assert result.cpus <= 4
    assert result.ram_limit == 16 * resources.GIB
    assert result.ram_available <= 15 * resources.GIB


def test_cgroup_parent_limit_and_usage_are_honoured(tmp_path, monkeypatch):
    parent = tmp_path / "job"
    parent.mkdir()
    (parent / "memory.max").write_text(str(8 * resources.GIB))
    (parent / "memory.current").write_text(str(3 * resources.GIB))
    (parent / "memory.stat").write_text(f"inactive_file {resources.GIB}\n")
    monkeypatch.delenv("SLURM_MEM_PER_NODE", raising=False)
    monkeypatch.delenv("SLURM_MEM_PER_CPU", raising=False)
    monkeypatch.setattr(resources, "cgroup_directories", lambda: [parent])
    result = resources.snapshot()
    assert result.ram_limit == 8 * resources.GIB
    assert result.ram_available <= 6 * resources.GIB


def test_per_cpu_memory_and_suffixes(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    monkeypatch.setenv("SLURM_MEM_PER_CPU", "2G")
    monkeypatch.delenv("SLURM_MEM_PER_NODE", raising=False)
    monkeypatch.setattr(resources, "cgroup_directories", list)
    assert resources.snapshot().ram_limit == 8 * resources.GIB
    assert resources.memory_bytes("16G") == 16 * resources.GIB
    assert resources.memory_bytes("16384") == 16 * resources.GIB


def test_visible_gpu_allocator_limit_and_headroom(monkeypatch):
    import sys
    from types import SimpleNamespace

    changes = []
    cuda = SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 1,
        mem_get_info=lambda device: (20 * resources.GIB, 24 * resources.GIB),
        get_per_process_memory_fraction=lambda device: 0.5,
        memory_reserved=lambda device: 3 * resources.GIB,
        get_device_properties=lambda device: SimpleNamespace(
            total_memory=24 * resources.GIB
        ),
        set_per_process_memory_fraction=lambda fraction, device: changes.append(
            (fraction, device)
        ),
    )
    torch = SimpleNamespace(cuda=cuda)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("CISEGMENTATION_GPU_MEMORY_LIMIT_MB", "12288")
    result = resources.snapshot()
    assert result.gpu_total == 12 * resources.GIB
    assert result.gpu_available == 9 * resources.GIB
    resources.apply_gpu_limit(torch)
    assert changes == [(0.5, 1)]
