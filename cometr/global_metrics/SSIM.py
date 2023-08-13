import torch
import torch.nn.functional as F
from torch import Tensor
from os.path import basename
import tempfile
import argparse
import numpy as np
import shutil
from beartype.typing import List, Optional, Tuple, Union
from beartype import beartype
from cometr.global_metrics.Metric import Metric


class SSIM(Metric):
    """Calculates the Structural Similarity Index (SSIM) between two HDF5 files containing voxel data."""

    @beartype
    def __init__(
        self,
        file1: str,
        file2: str,
        file1_key: str = "/entry/data/data",
        file2_key: str = "/entry/data/data",
        output_text: str = "output.txt",
        data_range: Optional[float] = None,
        kernel_size: Union[int, Tuple[int, int, int]] = 11,
        sigma: Union[float, Tuple[float, float, float]] = 1.5,
        K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
        device: Union[int, str] = "cpu",
        overwrite: bool = False,
        tmp_root: Optional[str] = None,
    ):
        super().__init__(file1, file2, file1_key, file2_key, output_text)

        # make into tuple
        self.kernel_size = kernel_size
        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size,) * 3
        self.sigma = sigma
        if isinstance(self.sigma, float):
            self.sigma = (self.sigma,) * 3
        self.K = K
        self.data_range = data_range
        self.device = device
        self.overwrite = overwrite
        self.tmp_root = tmp_root
        self.kernel_3d = None

    @beartype
    def gaussian_blur_3d(
        self,
        inp: torch.Tensor,
    ) -> Tensor:
        def gaussian(sz: int, sg: float, dev: Union[int, str] = "cpu") -> Tensor:
            g = torch.exp(
                -torch.pow(
                    torch.arange((1 - sz) / 2, (1 + sz) / 2, step=1, device=dev) / sg, 2
                )
                / 2
            )
            return (g / g.sum()).unsqueeze(dim=0)

        if self.kernel_3d is None:
            kernel_1d = gaussian(self.kernel_size[0], self.sigma[0], self.device)
            kernel_2d = torch.mul(
                torch.transpose(kernel_1d, 0, 1),
                gaussian(self.kernel_size[1], self.sigma[1], self.device),
            )
            self.kernel_3d = torch.mul(
                kernel_2d.unsqueeze(-1).repeat(1, 1, self.kernel_size[2]),
                gaussian(self.kernel_size[2], self.sigma[2], self.device).expand(
                    self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]
                ),
            )

        height_pad = (self.kernel_size[0] - 1) // 2
        width_pad = (self.kernel_size[1] - 1) // 2
        depth_pad = (self.kernel_size[2] - 1) // 2
        inp = F.pad(
            inp,
            (height_pad, height_pad, width_pad, width_pad, depth_pad, depth_pad),
            mode="reflect",
        )
        return F.conv3d(
            inp,
            self.kernel_3d.expand(
                inp.size(1),
                1,
                self.kernel_size[0],
                self.kernel_size[1],
                self.kernel_size[2],
            ),
            groups=inp.size(1),
        )

    @beartype
    def mu_sq(self, filename: str, file_key: str, dir_path: str) -> None:
        name = basename(filename)[:-3]

        A = torch.from_numpy(Metric.load_file(filename, file_key)).to(self.device)
        if A.dim() == 3:
            A = A.expand(1, 1, A.size(0), A.size(1), A.size(2))
        if A.dim() != 5:
            raise ValueError(
                "Data from the input filename has to 3D (HxWXD) or 5D (BxCxHxWxD)"
            )

        # This is mu
        A = self.gaussian_blur_3d(A)
        if self.device != "cpu":
            Metric.store_file(
                A.detach().cpu().numpy(),
                dir_path + "/" + name + "_mu.h5",
                file_key,
                self.overwrite,
            )
        else:
            Metric.store_file(
                A.detach().numpy(),
                dir_path + "/" + name + "_mu.h5",
                file_key,
                self.overwrite,
            )

        # This is mu_sq
        A = A.pow(2)
        if self.device != "cpu":
            Metric.store_file(
                A.detach().cpu().numpy(),
                dir_path + "/" + name + "_mu_sq.h5",
                file_key,
                self.overwrite,
            )
        else:
            Metric.store_file(
                A.detach().numpy(),
                dir_path + "/" + name + "_mu_sq.h5",
                file_key,
                self.overwrite,
            )

        A = torch.from_numpy(Metric.load_file(filename, file_key)).to(self.device)
        if A.dim() == 3:
            A = A.expand(1, 1, A.size(0), A.size(1), A.size(2))
        if A.dim() != 5:
            raise ValueError(
                "Data from the input filename has to 3D (HxWXD) or 5D (BxCxHxWxD)"
            )

        A = A.pow(2)

        A = self.gaussian_blur_3d(A)

        B = torch.from_numpy(
            Metric.load_file(dir_path + "/" + name + "_mu_sq.h5", file_key)
        ).to(self.device)
        if B.dim() == 3:
            B = B.expand(1, 1, B.size(0), B.size(1), B.size(2))
        if B.dim() != 5:
            raise ValueError(
                "Data from the input filename has to 3D (HxWXD) or 5D (BxCxHxWxD)"
            )

        A = A - B

        if self.device != "cpu":
            Metric.store_file(
                A.detach().cpu().numpy(),
                dir_path + "/" + name + "_sigma_sq.h5",
                file_key,
                self.overwrite,
            )
        else:
            Metric.store_file(
                A.detach().numpy(),
                dir_path + "/" + name + "_sigma_sq.h5",
                file_key,
                self.overwrite,
            )
        return

    @beartype
    def co(self, dir_path: str) -> None:
        name1 = basename(self.file1)[:-3]
        name2 = basename(self.file2)[:-3]

        # This is mu1
        A = torch.from_numpy(
            Metric.load_file(dir_path + "/" + name2 + "_mu.h5", self.file1_key)
        ).to(self.device)
        if A.dim() == 3:
            A = A.expand(1, 1, A.size(0), A.size(1), A.size(2))
        if A.dim() != 5:
            nm = dir_path + "/" + name2 + "_mu.h5"
            raise ValueError(f"Data in the {nm} has to 3D (HxWXD) or 5D (BxCxHxWxD)")

        B = torch.from_numpy(
            Metric.load_file(dir_path + "/" + name2 + "_mu.h5", self.file2_key)
        ).to(self.device)
        if B.dim() == 3:
            B = B.expand(1, 1, B.size(0), B.size(1), B.size(2))
        if B.dim() != 5:
            nm = dir_path + "/" + name2 + "_mu.h5"
            raise ValueError(f"Data in the {nm} has to 3D (HxWXD) or 5D (BxCxHxWxD)")

        # This is mu1_mu2
        A = A * B

        if self.device != "cpu":
            Metric.store_file(
                A.detach().cpu().numpy(),
                dir_path + "/" + name1 + "_" + name2 + "_mu1_mu2.h5",
                self.file1_key,
                self.overwrite,
            )
        else:
            Metric.store_file(
                A.detach().numpy(),
                dir_path + "/" + name1 + "_" + name2 + "_mu1_mu2.h5",
                self.file1_key,
                self.overwrite,
            )

        A = torch.from_numpy(Metric.load_file(self.file1, self.file1_key)).to(
            self.device
        )
        if A.dim() == 3:
            A = A.expand(1, 1, A.size(0), A.size(1), A.size(2))
        if A.dim() != 5:
            raise ValueError(
                f"Data in the {self.file1} has to 3D (HxWXD) or 5D (BxCxHxWxD)"
            )

        B = torch.from_numpy(Metric.load_file(self.file2, self.file2_key)).to(
            self.device
        )
        if B.dim() == 3:
            B = B.expand(1, 1, B.size(0), B.size(1), B.size(2))
        if B.dim() != 5:
            raise ValueError(
                f"Data in the {self.file2} has to 3D (HxWXD) or 5D (BxCxHxWxD)"
            )

        A = A * B

        A = self.gaussian_blur_3d(A)

        B = torch.from_numpy(
            Metric.load_file(
                dir_path + "/" + name1 + "_" + name2 + "_mu1_mu2.h5", self.file1_key
            )
        ).to(self.device)
        if B.dim() == 3:
            B = B.expand(1, 1, B.size(0), B.size(1), B.size(2))
        if B.dim() != 5:
            nm = dir_path + "/" + name1 + "_" + name2 + "_mu1_mu2.h5"
            raise ValueError(f"Data in the {nm} has to 3D (HxWXD) or 5D (BxCxHxWxD)")

        # This is sigma12
        A = A - B

        if self.device != "cpu":
            Metric.store_file(
                A.detach().cpu().numpy(),
                dir_path + "/" + name1 + "_" + name2 + "_sigma12.h5",
                self.file1_key,
                self.overwrite,
            )
        else:
            Metric.store_file(
                A.detach().numpy(),
                dir_path + "/" + name1 + "_" + name2 + "_sigma12.h5",
                self.file1_key,
                self.overwrite,
            )

        return

    @beartype
    def metric_calc(self, file1_data: np.ndarray, file2_data: np.ndarray) -> float:
        """Calculates the Structural Similarity index between two numpy arrays.

        Args:
            file1_data (np.ndarray): The numpy array containing voxel data from the first file.

            file2_data (np.ndarray): The numpy array containing voxel data from the second file.

        Returns:
            float: The mean squared error of the two numpy arrays.

        """

        K1, K2 = self.K
        if self.data_range is None:
            self.data_range = np.max(file2_data) - np.min(file2_data)
        C1 = (K1 * self.data_range) ** 2
        C2 = (K2 * self.data_range) ** 2

        dir_path = tempfile.mkdtemp(prefix=self.tmp_root)

        self.mu_sq(self.file1, self.file1_key, dir_path)

        self.mu_sq(self.file2, self.file2_key, dir_path)

        self.co(
            dir_path,
        )

        name1 = basename(self.file1)[:-3]
        name2 = basename(self.file2)[:-3]

        A = torch.from_numpy(
            Metric.load_file(dir_path + "/" + name1 + "_sigma_sq.h5", self.file1_key)
        ).to(self.device)
        if A.dim() == 3:
            A = A.expand(1, 1, A.size(0), A.size(1), A.size(2))
        if A.dim() != 5:
            nm = dir_path + "/" + name1 + "_sq.h5"
            raise ValueError(f"Data in the {nm} has to 3D (HxWXD) or 5D (BxCxHxWxD)")

        B = torch.from_numpy(
            Metric.load_file(dir_path + "/" + name2 + "_sigma_sq.h5", self.file2_key)
        ).to(self.device)
        if B.dim() == 3:
            B = B.expand(1, 1, B.size(0), B.size(1), B.size(2))
        if B.dim() != 5:
            nm = dir_path + "/" + name2 + "_sq.h5"
            raise ValueError(f"Data in the {nm} has to 3D (HxWXD) or 5D (BxCxHxWxD)")

        A = A + B + C2

        B = torch.from_numpy(
            Metric.load_file(
                dir_path + "/" + name1 + "_" + name2 + "_sigma12.h5", self.file1_key
            )
        ).to(self.device)
        if B.dim() == 3:
            B = B.expand(1, 1, B.size(0), B.size(1), B.size(2))
        if B.dim() != 5:
            nm = dir_path + "/" + name1 + "_" + name2 + "_csq.h5"
            raise ValueError(f"Data in the {nm} has to 3D (HxWXD) or 5D (BxCxHxWxD)")

        B = 2 * B + C2

        # This is cs_map
        A = B / A

        if self.device != "cpu":
            Metric.store_file(
                A.detach().cpu().numpy(),
                dir_path + "/" + name1 + "_" + name2 + "_cs_map.h5",
                self.file1_key,
                self.overwrite,
            )
        else:
            Metric.store_file(
                A.detach().numpy(),
                dir_path + "/" + name1 + "_" + name2 + "_cs_map.h5",
                self.file1_key,
                self.overwrite,
            )

        A = torch.from_numpy(
            Metric.load_file(dir_path + "/" + name1 + "_mu_sq.h5", self.file1_key)
        ).to(self.device)
        if A.dim() == 3:
            A = A.expand(1, 1, A.size(0), A.size(1), A.size(2))
        if A.dim() != 5:
            nm = dir_path + "/" + name1 + "_mu.h5"
            raise ValueError(f"Data in the {nm} has to 3D (HxWXD) or 5D (BxCxHxWxD)")

        B = torch.from_numpy(
            Metric.load_file(dir_path + "/" + name2 + "_mu_sq.h5", self.file2_key)
        ).to(self.device)
        if B.dim() == 3:
            B = B.expand(1, 1, B.size(0), B.size(1), B.size(2))
        if B.dim() != 5:
            nm = dir_path + "/" + name2 + "_mu_sq.h5"
            raise ValueError(f"Data in the {nm} has to 3D (HxWXD) or 5D (BxCxHxWxD)")

        A = A + B + C1

        B = torch.from_numpy(
            Metric.load_file(
                dir_path + "/" + name1 + "_" + name2 + "_mu1_mu2.h5", self.file1_key
            )
        ).to(self.device)
        if B.dim() == 3:
            B = B.expand(1, 1, B.size(0), B.size(1), B.size(2))
        if B.dim() != 5:
            nm = dir_path + "/" + name1 + "_" + name2 + "_mu1_mu2.h5"
            raise ValueError(f"Data in the {nm} has to 3D (HxWXD) or 5D (BxCxHxWxD)")

        B = 2 * B + C1

        A = B / A

        B = torch.from_numpy(
            Metric.load_file(
                dir_path + "/" + name1 + "_" + name2 + "_cs_map.h5", self.file1_key
            )
        ).to(self.device)
        if B.dim() == 3:
            B = B.expand(1, 1, B.size(0), B.size(1), B.size(2))
        if B.dim() != 5:
            nm = dir_path + "/" + name1 + "_" + name2 + "_cs_map.h5"
            raise ValueError(f"Data in the {nm} has to 3D (HxWXD) or 5D (BxCxHxWxD)")

        A = A * B

        shutil.rmtree(dir_path)

        A = torch.flatten(A, 2).mean(-1).mean()

        if self.device != "cpu":
            print(
                f"The Structural Similarity Index is:\n",
                A.detach().cpu().item(),
            )
            return A.detach().cpu().item()
        else:
            print(
                f"The Structural Similarity Index  is:\n",
                A.detach().item(),
            )
            return A.detach().item()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate the SSIM of two numpy arrays"
    )
    parser.add_argument("-f1", "--file1", required=True, help="Path to the first file")
    parser.add_argument("-f2", "--file2", required=True, help="Path to the second file")
    parser.add_argument(
        "-k1",
        "--file1_key",
        default="/entry/data/data",
        help="Key to data in the first file",
    )
    parser.add_argument(
        "-k2",
        "--file2_key",
        default="/entry/data/data",
        help="Key to data in the second file",
    )
    parser.add_argument(
        "-f3", "--output_text", default="output.txt", help="File to store result"
    )
    parser.add_argument(
        "-dr", "--data_range", default=None, help="Range of voxel values in file2"
    )
    parser.add_argument(
        "-kn", "--kernel_size", default=11, help="Size of the Gaussian 3D kernel"
    )
    parser.add_argument(
        "-sg", "--sigma", default=1.5, help="Sigma of the Gaussian 3D kernel"
    )
    parser.add_argument("-k", "--k_constants", default=(0.01, 0.03), help="K constants")
    parser.add_argument(
        "-d",
        "--device",
        default="cpu",
        help="Device where PyTorch will perform the calculations",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.set_defaults(overwrite=False)
    parser.add_argument(
        "-tmp",
        "--tmp_root",
        default=None,
        help="A prefix to be used when a temporary directory is created",
    )
    args = parser.parse_args()
    call_func = SSIM(
        args.file1,
        args.file2,
        args.file1_key,
        args.file2_key,
        args.output_text,
        args.data_range,
        args.kernel_size,
        args.sigma,
        args.k_constants,
        args.device,
        args.overwrite,
        args.tmp_root,
    ).calc()


if __name__ == "__main__":
    main()
