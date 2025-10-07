import unittest

from src.lib.domain import Domain, Url, parse_domain


class TestDomain(unittest.TestCase):
    def test_parse_domain(self):
        expected = Domain(subdomain="", domain_name="example", tld="com")
        actual = parse_domain(
            Url(
                scheme="http",
                netloc="example.com",
                path="",
                params="",
                query="",
                fragment="",
            )
        )
        self.assertEqual(expected, actual)

        expected = Domain(subdomain="", domain_name="10.10.10.10", tld="")
        actual = parse_domain(
            Url(
                scheme="http",
                netloc="10.10.10.10",
                path="/home",
                params="",
                query="a=b",
                fragment="",
            )
        )
        self.assertEqual(expected, actual)

        expected = Domain(subdomain="a.b", domain_name="example", tld="co.uk")
        actual = parse_domain(
            Url(
                scheme="https",
                netloc="a.b.example.co.uk",
                path="/",
                params="",
                query="",
                fragment="section1",
            )
        )
        self.assertEqual(expected, actual)

    def test_domain_host(self):
        domain = Domain(subdomain="a.b", domain_name="example", tld="co.uk")
        self.assertEqual("example.co.uk", domain.host)

        domain = Domain(subdomain="", domain_name="1.1.1.1", tld="")
        self.assertEqual("1.1.1.1", domain.host)
