

if __name__ == '__main__':
    import numpy as np
    from tqdm import tqdm
    from quantify_core.data.handling import (
        load_processed_dataset,
        get_tuids_containing,
    )
    from qce_interp.visualization.plot_logical_fidelity import get_fit_plot_arguments
    from file_manager.file_manager import (
        StorageManager,
        TUID,
        enforce_data_directory,
    )
    from qce_circuit.visualization.visualize_circuit.plotting_functionality import (
        construct_subplot,
        SubplotKeywordEnum,
        LabelFormat,
    )
    import matplotlib.pyplot as plt

    @enforce_data_directory
    def get_tuids():
        return get_tuids_containing(
            contains="Repeated stab meas 0 to 40 rounds ['D4', 'D1', 'D2'] qubits [0, 1, 0] state",
            t_start=TUID.datetime("20250526-200454-747-e499bf"),
            t_stop=TUID.datetime("20250526-224530-511-a39031"),
        )


    tuids = get_tuids()
    x_array: np.ndarray = np.asarray(load_processed_dataset(tuids[0], analysis_name="RepeatedStabilizerAnalysis")["qec_cycles"].values)
    defect_x_array: np.ndarray = np.arange(x_array[-1] + 1)
    defect_rates_z1: np.ndarray = np.zeros(shape=(len(tuids), len(defect_x_array)), dtype=float)
    post_selected_defect_rates_z1: np.ndarray = np.zeros(shape=(len(tuids), len(defect_x_array)), dtype=float)
    defect_rates_x1: np.ndarray = np.zeros(shape=(len(tuids), len(defect_x_array)), dtype=float)
    post_selected_defect_rates_x1: np.ndarray = np.zeros(shape=(len(tuids), len(defect_x_array)), dtype=float)
    mwpm_logical: np.ndarray = np.zeros(shape=(len(tuids), len(x_array)), dtype=float)

    for i, tuid in enumerate(tqdm(tuids, desc="Processing (even) defect rates")):
        defect_rates_z1[i, :] = np.asarray(load_processed_dataset(tuid, analysis_name="RepeatedStabilizerAnalysis")["defect_rates_Z1"].values)
        post_selected_defect_rates_z1[i, :] = np.asarray(load_processed_dataset(tuid, analysis_name="RepeatedStabilizerAnalysis")["defect_rates_post_selected_Z1"].values)
        defect_rates_x1[i, :] = np.asarray(load_processed_dataset(tuid, analysis_name="RepeatedStabilizerAnalysis")["defect_rates_X1"].values)
        post_selected_defect_rates_x1[i, :] = np.asarray(load_processed_dataset(tuid, analysis_name="RepeatedStabilizerAnalysis")["defect_rates_post_selected_X1"].values)
        mwpm_logical[i, :] = np.asarray(load_processed_dataset(tuid, analysis_name="RepeatedStabilizerAnalysis")["logical_fidelity_mwpm_d3_0"].values)

    kwargs = {
        SubplotKeywordEnum.LABEL_FORMAT.value: LabelFormat(
            x_label="QEC round",
            y_label="Defect rate $\\langle d_i \\rangle$"
        ),
        SubplotKeywordEnum.FIGURE_SIZE.value: (5.4, 4.8)
    }
    fig, (ax0, ax) = construct_subplot(nrows=2, gridspec_kw={"hspace": 0.2}, **kwargs)
    ax.plot(
        defect_x_array,
        defect_rates_z1.T,
        linestyle="-",
        marker="none",
        color="C0",
        alpha=0.1,
    )
    ax.plot(
        defect_x_array,
        defect_rates_x1.T,
        linestyle="-",
        marker="none",
        color="C2",
        alpha=0.1,
    )
    # Mean
    ax.plot(
        defect_x_array,
        np.mean(defect_rates_z1.T, axis=1),
        linestyle="-",
        marker=".",
        color="C0",
        alpha=1.0,
        label="Averaged defect rates Z1"
    )
    ax.plot(
        defect_x_array,
        np.mean(defect_rates_x1.T, axis=1),
        linestyle="-",
        marker=".",
        color="C2",
        alpha=1.0,
        label="Averaged defect rates X1"
    )

    ax0.plot(
        x_array,
        mwpm_logical.T,
        linestyle="-",
        marker="none",
        color="C8",
        alpha=0.1,
    )
    # Mean
    ax0.plot(
        x_array,
        np.mean(mwpm_logical.T, axis=1),
        linestyle="-",
        marker=".",
        color="C8",
        alpha=1.0,
        label="Averaged logical fidelity"
    )
    args, kwargs = get_fit_plot_arguments(x_array=x_array, y_array=np.mean(mwpm_logical.T, axis=1), exclude_first_n=3)
    ax0.errorbar(
        *args,
        yerr=0.0,
        **kwargs,
    )

    ax.set_ylim(0.0, 0.5)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax0.set_ylim(0.5, 1.0)
    ax0.legend(loc='upper left', bbox_to_anchor=(1, 1))

    fig.suptitle(f"Repetition code D4-D1-D2 (Z1-X1)\nts: 20250526-200454-747-e499bf to 20250526-224530-511-a39031")
    fig.tight_layout()
    fig.savefig(fname=StorageManager.figure_directory() / "defect_rate_statistics.png", transparent=False)
    plt.show()
